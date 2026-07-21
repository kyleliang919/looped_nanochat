"""Fully-controlled loop comparison: control BOTH parameters and compute.

Prior study matched compute/passes but not parameters (UT had 1 tied layer =
0.2M vs ours' 6 distinct layers = 1.19M). This fixes that with explicit variants:

  baseline      6 distinct layers, 1 pass                  | params P, compute 1x
  ours          6 distinct layers, 2 passes (routed)       | params P, compute 2x
  ut_isocompute 1 distinct layer, tied, looped 6x / 12x     | params P/6, compute 1x/2x
  ut_isoparam   6-distinct-layer GROUP, tied, looped 2x     | params P, compute 2x   <-- the clean iso-param, iso-compute match to ours
  ut_isoparam1  same 6-layer group, 1x (no loop)            | params P, compute 1x   (= a plain 6-layer net, sanity)

So the headline fair fight is ours vs ut_isoparam vs baseline: all same params P;
ours & ut_isoparam both 2x compute; baseline 1x.
All use warmup+cosine LR schedule. Task/config identical to loop_study.
"""
import argparse, math, json, time
import torch, torch.nn as nn, torch.nn.functional as F
import sys; sys.path.insert(0,'studies')
from loop_study import Block, rms
# Pre-committed task (chosen by baseline-only headroom screen, task_screen.py):
# modular cumulative sum with P=13, seqlen 64 — baseline ~0.32, chance 0.077, large headroom.
P=13; VOCAB=P
@torch.no_grad()
def gen_task(batch, seqlen, device, gen):
    d=torch.randint(0,P,(batch,seqlen),generator=gen,device=device)
    return d, torch.cumsum(d,1)%P

class FairGPT(nn.Module):
    def __init__(s,strat,d=128,h=4,n_layer=6,seqlen=64,gate_init=0.0):
        super().__init__(); s.strat=strat; s.n_layer=n_layer
        s.wte=nn.Embedding(VOCAB,d); s.pos=nn.Embedding(seqlen,d); s.head=nn.Linear(d,VOCAB,bias=False)
        if strat=='ut_isocompute':                       # 1 physical layer, tied loop to match 1x effective depth (=n_layer)
            s.blocks=nn.ModuleList([Block(d,h)]); s.K=n_layer
        elif strat in ('ut_isocompute2',):               # tied single layer, 2x effective depth
            s.blocks=nn.ModuleList([Block(d,h)]); s.K=2*n_layer
        else:                                            # baseline / ours / ut_isoparam*: n_layer distinct layers
            s.blocks=nn.ModuleList([Block(d,h) for _ in range(n_layer)])
        if strat.startswith('ours'):
            s.gate_logit=nn.Parameter(torch.full((n_layer,),float(gate_init)))
            s.src=max(0,n_layer//2-1)                    # source layer (mid for deep, first for shallow)
    def forward(s,idx):
        B,T=idx.shape; emb=s.wte(idx)+s.pos(torch.arange(T,device=idx.device))[None]; x=emb; st=s.strat
        if st=='baseline':
            for blk in s.blocks: x,_=blk(x)
        elif st in ('ut_isocompute','ut_isocompute2'):    # tied single layer looped K
            blk=s.blocks[0]
            for _ in range(s.K): x,_=blk(x)
        elif st=='ut_isoparam1':                          # the 6-layer group, once (plain 6-layer net)
            for blk in s.blocks: x,_=blk(x)
        elif st=='ut_isoparam':                           # 6-layer group, tied, looped 2x (== ours' params & compute)
            for _ in range(2):
                for blk in s.blocks: x,_=blk(x)
        else:                                             # ours: 6 layers, pass1 caches src q/k, pass2 routes
            x1=emb; cache=None
            for i,blk in enumerate(s.blocks):
                x1,qk=blk(x1)
                if i==s.src: cache=qk
            x2=emb
            for i,blk in enumerate(s.blocks):
                g=F.softplus(s.gate_logit[i]); x2,_=blk(x2,route_qk=cache,gate=g)
            x=x2
        return s.head(rms(x))
    def passes(s): return {'baseline':1,'ut_isocompute':1,'ut_isocompute2':2,'ut_isoparam1':1,'ut_isoparam':2,'ours':2}[s.strat]

def np_(m): return sum(p.numel() for p in m.parameters())

def run(strat,seed,lr,steps,dev,d=128,h=4,n_layer=6,seqlen=64,bs=256):
    torch.manual_seed(seed)
    g=torch.Generator(device=dev); g.manual_seed(seed+1000); vg=torch.Generator(device=dev); vg.manual_seed(99999)
    m=FairGPT(strat,d,h,n_layer,seqlen).to(dev)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,betas=(0.9,0.95),weight_decay=0.01)
    warm=max(1,int(0.1*steps))
    def lr_at(t):
        if t<warm: return lr*(t+1)/warm
        p=(t-warm)/max(1,steps-warm); return lr*(0.05+0.95*0.5*(1+math.cos(math.pi*p)))
    vin,vt=gen_task(256,seqlen,dev,vg); gns=[]; curve=[]; diverged=False
    for step in range(steps):
        for pg in opt.param_groups: pg['lr']=lr_at(step)
        din,tgt=gen_task(bs,seqlen,dev,g)
        loss=F.cross_entropy(m(din).reshape(-1,VOCAB),tgt.reshape(-1))
        opt.zero_grad(); loss.backward()
        gns.append(torch.nn.utils.clip_grad_norm_(m.parameters(),1e9).item())
        if not math.isfinite(loss.item()) or loss.item()>50: diverged=True; break
        opt.step()
        if step%max(1,steps//20)==0 or step==steps-1:
            with torch.no_grad(): curve.append((m(vin).argmax(-1)==vt).float().mean().item())
    return dict(strat=strat,seed=seed,lr=lr,passes=m.passes(),params=np_(m),diverged=diverged,
        final_acc=(curve[-1] if curve and not diverged else 0.0), gmax=float(max(gns)) if gns else 0.0)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--steps',type=int,default=1500); ap.add_argument('--seeds',type=int,default=3)
    ap.add_argument('--n_layer',type=int,default=6)
    ap.add_argument('--out',type=str,default='studies/fair_results.json'); a=ap.parse_args(); dev='cuda'
    lrs=[3e-4,1e-3,3e-3]
    strats=['baseline','ut_isocompute','ut_isocompute2','ut_isoparam1','ut_isoparam','ours']
    res=[]; t0=time.time()
    for st in strats:
        for lr in lrs:
            for seed in range(a.seeds):
                r=run(st,seed,lr,a.steps,dev,n_layer=a.n_layer); r['n_layer']=a.n_layer; res.append(r)
                print(f"L{a.n_layer} {st:14s} lr={lr:.0e} s{seed} | acc={r['final_acc']:.3f} gmax={r['gmax']:.0f} "
                      f"{r['passes']}x {r['params']/1e6:.2f}M {'DIV' if r['diverged'] else ''}",flush=True)
    json.dump(res,open(a.out,'w')); print(f"\ndone {(time.time()-t0)/60:.1f} min -> {a.out}")
