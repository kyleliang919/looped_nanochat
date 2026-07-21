"""Screen candidate tasks by BASELINE-ONLY accuracy to find one with real headroom.

Headroom = baseline lands clearly below ceiling but above chance. That gap is the
room a loop could fill. We pick the task/difficulty by baseline alone (no loop
models involved), THEN commit to it for the fair loop comparison — so it isn't
cherry-picked for a favorable loop result.

Tasks (all favor sequential/state-tracking computation that shallow nets struggle with):
  parity   : running prefix-XOR of a bit string           (chance 0.5)
  cumsum   : running sum mod P                             (chance 1/P)
  permcomp : track composition of permutations of {0..n-1}, predict image of a fixed element (chance 1/n)
  recall   : associative recall — last value bound to a queried key (chance 1/V)
Difficulty knobs: sequence length, modulus/alphabet, #layers.
"""
import argparse, math, json, time
import torch, torch.nn as nn, torch.nn.functional as F
import sys; sys.path.insert(0,'studies')
from loop_study import Block, rms

def make_batch(task,bs,seqlen,gen,dev,P=7,n=5,V=16):
    if task=='parity':
        x=torch.randint(0,2,(bs,seqlen),generator=gen,device=dev)
        y=torch.cumsum(x,1)%2; return x,y,2
    if task=='cumsum':
        x=torch.randint(0,P,(bs,seqlen),generator=gen,device=dev)
        y=torch.cumsum(x,1)%P; return x,y,P
    if task=='permcomp':
        # each token is a permutation of {0..n-1} drawn from a small generator set (adjacent transpositions + identity)
        # state = running composition; target = image of element 0 under the running composition.
        gens=[list(range(n))]  # identity
        for i in range(n-1):
            p=list(range(n)); p[i],p[i+1]=p[i+1],p[i]; gens.append(p)  # adjacent swaps
        G=torch.tensor(gens,device=dev)                      # (n_gen, n)
        idx=torch.randint(0,len(gens),(bs,seqlen),generator=gen,device=dev)  # token = which generator
        # compose left-to-right; track image of 0
        cur=torch.zeros(bs,dtype=torch.long,device=dev)      # image of element 0, starts at 0
        ys=[]
        for t in range(seqlen):
            perm=G[idx[:,t]]                                 # (bs,n)
            cur=perm.gather(1,cur.unsqueeze(1)).squeeze(1)   # apply perm to current image
            ys.append(cur)
        y=torch.stack(ys,1); return idx,y,n
    if task=='recall':
        # tokens alternate key,value,key,value... ; at each value position target = value most recently bound to CURRENT key...
        # simpler: sequence of (key,val) pairs then a query key; predict its bound value. We do per-position: at each t, predict
        # the value last seen for key x[t] (0 if unseen). Vocab split: keys 0..V-1 as inputs, predict value 0..V-1.
        keys=torch.randint(0,V,(bs,seqlen),generator=gen,device=dev)
        vals=torch.randint(0,V,(bs,seqlen),generator=gen,device=dev)
        # target[t] = val at the most recent earlier position with same key, else 0
        y=torch.zeros(bs,seqlen,dtype=torch.long,device=dev)
        last={}
        # vectorized-ish over t (seqlen small)
        lastval=torch.full((bs,V),-1,dtype=torch.long,device=dev)
        ar=torch.arange(bs,device=dev)
        for t in range(seqlen):
            k=keys[:,t]; prev=lastval[ar,k]
            y[:,t]=torch.where(prev>=0,prev,torch.zeros_like(prev))
            lastval[ar,k]=vals[:,t]
        # input embeds both key and val: fold as key*V+val into one token id (vocab V*V)
        x=keys*V+vals; return x,y,V  # predict value 0..V-1
    raise ValueError(task)

class Baseline(nn.Module):
    def __init__(s,vin,vout,d=128,h=4,n_layer=6,seqlen=48):
        super().__init__(); s.wte=nn.Embedding(vin,d); s.pos=nn.Embedding(seqlen,d)
        s.blocks=nn.ModuleList([Block(d,h) for _ in range(n_layer)]); s.head=nn.Linear(d,vout,bias=False)
    def forward(s,x):
        h=s.wte(x)+s.pos(torch.arange(x.size(1),device=x.device))[None]
        for b in s.blocks: h,_=b(h)
        return s.head(rms(h))

def screen(task,seqlen,n_layer,steps,dev,P=7,n=5,V=16,lr=1e-3,seed=0):
    torch.manual_seed(seed); g=torch.Generator(device=dev); g.manual_seed(seed+1); vg=torch.Generator(device=dev); vg.manual_seed(999)
    # infer vocab sizes
    vin={'parity':2,'cumsum':P,'permcomp':n,'recall':V*V}[task]
    vout={'parity':2,'cumsum':P,'permcomp':n,'recall':V}[task]
    m=Baseline(vin,vout,n_layer=n_layer,seqlen=seqlen).to(dev)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,betas=(0.9,0.95),weight_decay=0.01)
    warm=max(1,steps//10)
    def lrat(t):
        if t<warm: return lr*(t+1)/warm
        p=(t-warm)/max(1,steps-warm); return lr*(0.05+0.95*0.5*(1+math.cos(math.pi*p)))
    vx,vy,_=make_batch(task,512,seqlen,vg,dev,P,n,V)
    for t in range(steps):
        for pg in opt.param_groups: pg['lr']=lrat(t)
        x,y,_=make_batch(task,256,seqlen,g,dev,P,n,V)
        loss=F.cross_entropy(m(x).reshape(-1,vout),y.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    with torch.no_grad():
        acc=(m(vx).argmax(-1)==vy).float().mean().item()
    chance={'parity':0.5,'cumsum':1/P,'permcomp':1/n,'recall':1/V}[task]
    return acc,chance

if __name__=='__main__':
    dev='cuda'; steps=1200
    print(f"{'task':10s}{'config':26s}{'baseline acc':>13}{'chance':>8}{'headroom?':>11}")
    trials=[
        ('parity',dict(seqlen=64,n_layer=4)),
        ('parity',dict(seqlen=96,n_layer=4)),
        ('cumsum',dict(seqlen=48,n_layer=4,P=11)),
        ('cumsum',dict(seqlen=64,n_layer=4,P=13)),
        ('permcomp',dict(seqlen=48,n_layer=4,n=5)),
        ('permcomp',dict(seqlen=64,n_layer=4,n=5)),
        ('permcomp',dict(seqlen=48,n_layer=2,n=5)),
        ('recall',dict(seqlen=48,n_layer=4,V=16)),
        ('recall',dict(seqlen=64,n_layer=4,V=24)),
    ]
    for task,cfg in trials:
        acc,ch=screen(task,steps=steps,dev=dev,**cfg)
        gap=acc-ch; hr = 'YES' if (acc<0.85 and acc-ch>0.1) else ('saturated' if acc>=0.85 else 'near-chance')
        print(f"{task:10s}{str(cfg):26s}{acc:>13.3f}{ch:>8.3f}{hr:>11}",flush=True)
