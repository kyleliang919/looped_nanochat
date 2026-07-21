"""Does TRAINING our routing with more than 2 passes help? (architectural K-pass extension)

K-pass additive routing: pass 0 runs raw and caches (q,k) at the source layer;
each subsequent pass re-runs from embeddings, adds the additive attention-logit
prior from the PREVIOUS pass's cache, and caches its own. K=2 == the shipped design.
Trained end-to-end (full backprop through all K passes). Same toy task/config as
loop_study, so numbers are comparable.
"""
import argparse, math, json, time
import torch, torch.nn as nn, torch.nn.functional as F
import sys; sys.path.insert(0,'studies')
from loop_study import gen_task, VOCAB, Block, rms, n_params

class KPassGPT(nn.Module):
    def __init__(s,K,d=128,h=4,n_layer=6,seqlen=48,gate_init=0.0):
        super().__init__(); s.K=K; s.n_layer=n_layer
        s.wte=nn.Embedding(VOCAB,d); s.pos=nn.Embedding(seqlen,d); s.head=nn.Linear(d,VOCAB,bias=False)
        s.blocks=nn.ModuleList([Block(d,h) for _ in range(n_layer)])
        # a gate per (routed-pass, layer): passes 1..K-1 each route
        s.gate_logit=nn.Parameter(torch.full((K-1,n_layer),float(gate_init)))
        s.src=n_layer//2
    def forward(s,idx):
        B,T=idx.shape; emb=s.wte(idx)+s.pos(torch.arange(T,device=idx.device))[None]
        # pass 0: raw, cache q/k at source
        x=emb; cache=None
        for i,blk in enumerate(s.blocks):
            x,qk=blk(x)
            if i==s.src: cache=qk
        # passes 1..K-1: route from previous pass's cache, re-cache own
        for p in range(1,s.K):
            x=emb; newcache=None
            for i,blk in enumerate(s.blocks):
                g=F.softplus(s.gate_logit[p-1,i])
                x,qk=blk(x,route_qk=cache,gate=g)
                if i==s.src: newcache=qk
            cache=newcache
        return s.head(rms(x))

def run(K,seed,lr,steps,dev,d=128,h=4,n_layer=6,seqlen=48,bs=256):
    torch.manual_seed(seed)
    g=torch.Generator(device=dev); g.manual_seed(seed+1000)
    vg=torch.Generator(device=dev); vg.manual_seed(99999)
    m=KPassGPT(K,d,h,n_layer,seqlen).to(dev)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,betas=(0.9,0.95),weight_decay=0.01)
    vin,vt=gen_task(256,seqlen,dev,vg)
    # LR schedule: linear warmup then cosine decay — the standard remedy that deep/
    # recurrent loops effectively require; a constant LR is hostile to deep BPTT.
    warm=max(1,int(0.1*steps))
    def lr_at(t):
        if t<warm: return lr*(t+1)/warm
        p=(t-warm)/max(1,steps-warm); return lr*(0.05+0.95*0.5*(1+math.cos(math.pi*p)))
    gns=[]; curve=[]; diverged=False
    for step in range(steps):
        for pg in opt.param_groups: pg['lr']=lr_at(step)
        din,tgt=gen_task(bs,seqlen,dev,g)
        loss=F.cross_entropy(m(din).reshape(-1,VOCAB),tgt.reshape(-1))
        opt.zero_grad(); loss.backward()
        gns.append(torch.nn.utils.clip_grad_norm_(m.parameters(),1e9).item())
        if not math.isfinite(loss.item()) or loss.item()>50: diverged=True; break
        opt.step()
        if step%max(1,steps//20)==0 or step==steps-1:
            with torch.no_grad():
                acc=(m(vin).argmax(-1)==vt).float().mean().item()
            curve.append(acc)
    return dict(K=K,seed=seed,lr=lr,passes=K,params=n_params(m),diverged=diverged,
        final_acc=(curve[-1] if curve and not diverged else 0.0),
        gmax=float(max(gns)) if gns else 0.0)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--steps',type=int,default=1500); ap.add_argument('--seeds',type=int,default=3)
    ap.add_argument('--Ks',type=str,default='2,3,4')
    ap.add_argument('--out',type=str,default='studies/kpass_results.json'); a=ap.parse_args(); dev='cuda'
    lrs=[1e-4,3e-4,1e-3,3e-3]  # extend downward for deep loops; +warmup/cosine schedule (see run())
    res=[]; t0=time.time()
    for K in [int(k) for k in a.Ks.split(',')]:
        for lr in lrs:
            for seed in range(a.seeds):
                r=run(K,seed,lr,a.steps,dev); res.append(r)
                print(f"K={K} lr={lr:.0e} s{seed} | acc={r['final_acc']:.3f} gmax={r['gmax']:.0f} passes={r['passes']}x "
                      f"params={r['params']/1e6:.2f}M {'DIV' if r['diverged'] else ''}",flush=True)
    json.dump(res,open(a.out,'w')); print(f"\ndone {(time.time()-t0)/60:.1f} min -> {a.out}")
