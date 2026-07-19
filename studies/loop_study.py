"""
Toy-scale controlled study: which loop-transformer strategy trains best & most stably?

FAIR-COMPARISON contract (held fixed across every strategy):
  same vocab / d_model / n_head / seqlen / batch / optimizer / #steps / data stream / seeds.
  We additionally report forward-pass count (compute multiplier) so quality is read
  per-compute, never hiding that "more loops = more FLOPs".

Task: modular cumulative sum — predict (sum of inputs[0..t]) mod 7 at each position.
A long associative scan that rewards extra depth/iteration and does not saturate at
toy scale, so strategies actually separate. Fully synthetic, GPU-vectorized, reproducible.
(An earlier Dyck-2 variant saturated too fast to discriminate; see git history.)

Strategies (grounded in the loop-transformer literature):
  baseline    1x  single-pass GPT (control)
  ut2 / ut4   2x/4x  Universal Transformer: weight-tied block looped K, FULL BPTT   [Dehghani 2019]
  ut4_tbptt   4x  weight-tied K=4 loop, TRUNCATED BPTT (grad through last iter only) [Tallec2017; Geiping2025]
  ut4_inject  4x  weight-tied K=4 with input injection each step (recurrent-depth)   [Geiping 2025]
  act         ~   UT with learned ACT halting (ponder), time-penalty loss           [Graves 2016]
  deq         4x  weight-tied fixed-point iterate, grad through last step only (DEQ-ish) [Bai 2019]
  ours        2x  two-pass additive attention-logit routing, gate=softplus init~0 (warm start)
  ours_hot    2x  same but gate init LARGE (~1) — removes the warm start (ablation)

Metrics per (strategy, seed, lr): final/best val loss, closing-bracket accuracy,
grad-norm median/max, divergence flag, late-training loss std (stability).
"""
import argparse, math, json, time
import torch, torch.nn as nn, torch.nn.functional as F

# ---------------- task: modular cumulative sum (associative scan; depth/iteration helps) ----------------
# Input digits 0..P-1; target at each position t = (sum of inputs[0..t]) mod P.
# Requires composing a long associative scan -> rewards extra compute/depth; no saturation at toy scale.
P=7; VOCAB=P
@torch.no_grad()
def gen_task(batch, seqlen, device, gen):
    d=torch.randint(0,P,(batch,seqlen),generator=gen,device=device)
    tgt=torch.cumsum(d,1)%P
    return d,tgt

# ---------------- model pieces ----------------
def rms(x): return F.rms_norm(x,(x.size(-1),))
class Attn(nn.Module):
    def __init__(s,d,h):
        super().__init__(); s.h=h; s.hd=d//h
        s.qkv=nn.Linear(d,3*d,bias=False); s.o=nn.Linear(d,d,bias=False)
    def forward(s,x,route_qk=None,gate=None):
        B,T,D=x.shape
        q,k,v=s.qkv(x).split(D,2)
        q=q.view(B,T,s.h,s.hd).transpose(1,2); k=k.view(B,T,s.h,s.hd).transpose(1,2); v=v.view(B,T,s.h,s.hd).transpose(1,2)
        sc=1.0/math.sqrt(s.hd); logits=(q@k.transpose(-2,-1))*sc
        if route_qk is not None:
            rq,rk=route_qk; logits=logits+gate*((rq@rk.transpose(-2,-1))*sc)
        m=torch.triu(torch.ones(T,T,device=x.device,dtype=torch.bool),1)
        y=(F.softmax(logits.masked_fill(m,float('-inf')),-1)@v).transpose(1,2).contiguous().view(B,T,D)
        return s.o(y),(q,k)
class MLP(nn.Module):
    def __init__(s,d):
        super().__init__(); s.f=nn.Linear(d,4*d,bias=False); s.p=nn.Linear(4*d,d,bias=False)
    def forward(s,x): return s.p(F.relu(s.f(x)).square())
class Block(nn.Module):
    def __init__(s,d,h):
        super().__init__(); s.a=Attn(d,h); s.m=MLP(d)
    def forward(s,x,route_qk=None,gate=None):
        y,qk=s.a(rms(x),route_qk,gate); x=x+y; return x+s.m(rms(x)),qk

class GPT(nn.Module):
    def __init__(s,strat,d=256,h=4,n_layer=6,seqlen=128,gate_init=0.0):
        super().__init__(); s.strat=strat; s.n_layer=n_layer
        s.wte=nn.Embedding(VOCAB,d); s.pos=nn.Embedding(seqlen,d); s.head=nn.Linear(d,VOCAB,bias=False)
        tied = strat in ('ut2','ut4','ut4_tbptt','ut4_inject','act','deq')
        if tied:
            s.blocks=nn.ModuleList([Block(d,h)])          # 1 physical block, weight-tied loop
            s.K={'ut2':2,'ut4':4,'ut4_tbptt':4,'ut4_inject':4,'act':4,'deq':6}[strat]
        else:
            s.blocks=nn.ModuleList([Block(d,h) for _ in range(n_layer)]); s.K=1
        if strat=='act':
            s.halt=nn.Linear(d,1)                          # ACT halting unit
        if strat.startswith('ours'):
            s.gate_logit=nn.Parameter(torch.full((n_layer,),float(gate_init))); s.src=n_layer//2
    def fwd_passes(s):  # forward-pass compute multiplier (for per-compute fairness)
        return {'baseline':1,'ut2':2,'ut4':4,'ut4_tbptt':4,'ut4_inject':4,'act':4,'deq':6,
                'ours':2,'ours_hot':2}.get(s.strat,1)
    def forward(s,idx):
        B,T=idx.shape; emb=s.wte(idx)+s.pos(torch.arange(T,device=idx.device))[None]; x=emb
        st=s.strat
        if st=='baseline':
            for blk in s.blocks: x,_=blk(x)
        elif st in ('ut2','ut4'):                          # full BPTT through K tied iters
            blk=s.blocks[0]
            for _ in range(s.K): x,_=blk(x)
        elif st=='ut4_tbptt':                              # truncated BPTT: detach all but last iter
            blk=s.blocks[0]
            for i in range(s.K):
                if i<s.K-1:
                    with torch.no_grad(): x,_=blk(x)
                    x=x.detach()
                else: x,_=blk(x)
        elif st=='ut4_inject':                             # recurrent-depth: re-inject embeddings
            blk=s.blocks[0]
            for _ in range(s.K): x,_=blk(x+emb)
        elif st=='deq':                                    # DEQ-ish: iterate to ~fixed point, grad last step only
            blk=s.blocks[0]
            with torch.no_grad():
                for _ in range(s.K-1): x,_=blk(x+emb)
            x=x.detach(); x,_=blk(x+emb)                   # one grad-carrying step from (approx) equilibrium
        elif st=='act':                                    # ACT halting
            blk=s.blocks[0]; halt_cum=torch.zeros(B,T,1,device=idx.device); out=torch.zeros_like(x); rem=torch.ones(B,T,1,device=idx.device)
            s.ponder=torch.zeros((),device=idx.device)
            for i in range(s.K):
                x,_=blk(x); p=torch.sigmoid(s.halt(rms(x)))
                if i==s.K-1: p=torch.ones_like(p)
                w=torch.minimum(p,rem); out=out+w*x; s.ponder=s.ponder+rem.mean(); rem=(rem-w).clamp(min=0)
            x=out
        else:                                              # ours: two-pass additive routing
            x1=emb; cache=None
            for i,blk in enumerate(s.blocks):
                x1,qk=blk(x1)
                if i==s.src: cache=qk
            x2=emb
            for i,blk in enumerate(s.blocks):
                g=F.softplus(s.gate_logit[i]); x2,_=blk(x2,route_qk=cache,gate=g)
            x=x2
        return s.head(rms(x))
def n_params(m): return sum(p.numel() for p in m.parameters())

# ---------------- train/eval ----------------
def run(strat,seed,lr,steps,device,d=256,h=4,n_layer=6,seqlen=128,bs=64,gate_init=0.0):
    torch.manual_seed(seed)
    g=torch.Generator(device=device); g.manual_seed(seed+1000)
    vg=torch.Generator(device=device); vg.manual_seed(99999)
    m=GPT(strat,d=d,h=h,n_layer=n_layer,seqlen=seqlen,gate_init=gate_init).to(device)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,betas=(0.9,0.95),weight_decay=0.01)
    vin,vt=gen_task(256,seqlen,device,vg)
    curve=[]; gns=[]; diverged=False
    for step in range(steps):
        din,tgt=gen_task(bs,seqlen,device,g)
        logits=m(din)
        loss=F.cross_entropy(logits.reshape(-1,VOCAB),tgt.reshape(-1))
        if strat=='act': loss=loss+1e-3*m.ponder            # ACT time penalty
        opt.zero_grad(); loss.backward()
        gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1e9).item(); gns.append(gn)  # measure only
        if not math.isfinite(loss.item()) or loss.item()>50: diverged=True; break
        opt.step()
        if step%max(1,steps//40)==0 or step==steps-1:
            with torch.no_grad():
                vl=m(vin)
                vloss=F.cross_entropy(vl.reshape(-1,VOCAB),vt.reshape(-1)).item()
                acc=(vl.argmax(-1)==vt).float().mean().item()
            curve.append((step,vloss,acc))
    fin=[l for _,l,_ in curve if math.isfinite(l)]; late=[l for _,l,_ in curve[-8:] if math.isfinite(l)]
    return dict(config=strat,strat=strat,seed=seed,lr=lr,params=n_params(m),passes=m.fwd_passes(),diverged=diverged,
        final_val=(curve[-1][1] if curve and not diverged else float('inf')),
        final_acc=(curve[-1][2] if curve and not diverged else 0.0),
        best_val=(min(fin) if fin else float('inf')),
        late_std=(float(torch.tensor(late).std()) if len(late)>1 else 0.0),
        gradnorm_p50=float(torch.tensor(gns).median()) if gns else 0.0,
        gradnorm_max=float(max(gns)) if gns else 0.0, curve=curve)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument("--steps",type=int,default=1500); ap.add_argument('--seeds',type=int,default=3)
    ap.add_argument('--out',type=str,default='studies/results.json'); ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args(); dev='cuda'
    lrs=[1e-3,3e-3,1e-2,3e-2]  # incl. aggressive LR to expose instability
    configs=[('baseline',0.0),('ut2',0.0),('ut4',0.0),('ut4_tbptt',0.0),('ut4_inject',0.0),
             ('act',0.0),('deq',0.0),('ours',0.0),('ours_hot',1.0)]
    if a.smoke: lrs=[3e-3]; a.seeds=1; a.steps=30
    results=[]; t0=time.time()
    for cfg,gi in configs:
        real='ours' if cfg=='ours_hot' else cfg
        for lr in lrs:
            for seed in range(a.seeds):
                r=run(real,seed,lr,a.steps,dev,d=128,h=4,n_layer=6,seqlen=48,bs=256,gate_init=gi); r['config']=cfg; results.append(r)
                print(f"{cfg:11s} lr={lr:.0e} s{seed} | val={r['final_val']:.3f} acc={r['final_acc']:.3f} "
                      f"gmax={r['gradnorm_max']:.0f} passes={r['passes']}x {'DIV' if r['diverged'] else ''}",flush=True)
    json.dump(results,open(a.out,'w')); print(f"\ndone {(time.time()-t0)/60:.1f} min -> {a.out}")
