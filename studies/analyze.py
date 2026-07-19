"""Aggregate loop_study results into a quality + stability leaderboard."""
import json, sys, math
from collections import defaultdict
res=json.load(open(sys.argv[1] if len(sys.argv)>1 else 'studies/results.json'))

ORDER=['baseline','ut2','ut4','ut4_tbptt','ut4_inject','act','deq','ours','ours_hot']
by=defaultdict(list)
for r in res: by[r['config']].append(r)

def stats(vals):
    vals=[v for v in vals if math.isfinite(v)]
    if not vals: return (float('inf'),0.0)
    m=sum(vals)/len(vals); sd=(sum((v-m)**2 for v in vals)/len(vals))**0.5
    return m,sd

print(f"{'strategy':11s} {'passes':>6} | {'acc(best-LR)':>12} {'acc mean±sd':>16} | "
      f"{'divg%':>6} {'gmax p90':>9} {'seedσ@bestLR':>13}")
print('-'*92)
rows=[]
for cfg in ORDER:
    rs=by.get(cfg,[])
    if not rs: continue
    passes=rs[0]['passes']
    # group by lr
    bylr=defaultdict(list)
    for r in rs: bylr[r['lr']].append(r)
    # best-LR = lr with highest mean acc among non-diverged
    lr_acc={lr:stats([x['final_acc'] for x in g if not x['diverged']])[0] for lr,g in bylr.items()}
    best_lr=max(lr_acc,key=lr_acc.get)
    bestg=bylr[best_lr]
    best_acc=max(x['final_acc'] for x in bestg)
    acc_m,acc_sd=stats([x['final_acc'] for x in rs if not x['diverged']])
    divg=100*sum(1 for x in rs if x['diverged'])/len(rs)
    gmax=sorted(x['gradnorm_max'] for x in rs)
    gmax_p90=gmax[int(len(gmax)*0.9)] if gmax else 0
    seed_sd=stats([x['final_acc'] for x in bestg if not x['diverged']])[1]
    rows.append((cfg,passes,best_acc,acc_m,acc_sd,divg,gmax_p90,seed_sd,best_lr))
    print(f"{cfg:11s} {passes:>5}x | {best_acc:>12.3f} {acc_m:>7.3f}±{acc_sd:<7.3f} | "
          f"{divg:>5.0f}% {gmax_p90:>9.0f} {seed_sd:>13.3f}  (bestLR={best_lr:.0e})")

print("\nInterpretation axes:")
print(" - QUALITY: acc(best-LR) — peak achievable; higher = better")
print(" - STABILITY: divg% (lower=better), gmax p90 (grad spikes; lower=better),")
print("             seedσ@bestLR (run-to-run variance; lower=more reliable),")
print("             acc mean±sd across ALL LRs (LR-robustness; higher mean + lower sd = more forgiving)")

# LR-robustness: how much does acc drop at the most aggressive LR vs best?
print("\nLR sensitivity (mean acc by LR, non-diverged):")
lrs=sorted({r['lr'] for r in res})
hdr='strategy    '+''.join(f"{lr:>9.0e}" for lr in lrs)
print(hdr); print('-'*len(hdr))
for cfg in ORDER:
    rs=by.get(cfg,[])
    if not rs: continue
    line=f"{cfg:11s} "
    for lr in lrs:
        g=[x['final_acc'] for x in rs if x['lr']==lr and not x['diverged']]
        line+=f"{(sum(g)/len(g) if g else float('nan')):>9.3f}"
    print(line)
