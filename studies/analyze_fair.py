"""Analyze the iso-param / iso-compute fair comparison (per depth)."""
import json,sys
from collections import defaultdict
def st(v): v=[a for a in v if a==a]; m=sum(v)/len(v); return m,(sum((a-m)**2 for a in v)/len(v))**0.5 if len(v)>1 else 0
NAMES={'baseline':'baseline (1x)','ut_isocompute':'UT iso-compute (tied 1-layer)',
       'ut_isocompute2':'UT iso-compute 2x','ut_isoparam1':'6-layer group x1 (=deep baseline)',
       'ut_isoparam':'UT iso-param (tied group x2)','ours':'ours (routed x2)'}
for f in sys.argv[1:]:
    res=json.load(open(f)); L=res[0].get('n_layer','?')
    by=defaultdict(list)
    for r in res: by[r['strat']].append(r)
    print(f"\n=== {f}  (base layers L={L}) ===")
    print(f"{'strategy':32s}{'params':>8}{'passes':>7}{'mean acc':>9}{'range':>13}{'σ':>7}{'gmax p90':>9}")
    for k in ['baseline','ut_isocompute','ut_isocompute2','ut_isoparam1','ut_isoparam','ours']:
        rs=by.get(k,[])
        if not rs: continue
        bylr=defaultdict(list)
        for r in rs:
            if not r['diverged']: bylr[r['lr']].append(r['final_acc'])
        blr=max(bylr,key=lambda l:st(bylr[l])[0]); accs=sorted(bylr[blr]); m,sd=st(accs)
        gp=sorted(x['gmax'] for x in rs)[int(len(rs)*0.9)]
        print(f"{NAMES[k]:32s}{rs[0]['params']/1e6:>7.2f}M{rs[0]['passes']:>6}x{m:>9.3f}{min(accs):>6.2f}-{max(accs):<6.2f}{sd:>7.3f}{gp:>9.0f}")
    print("Fair fight = ours vs UT iso-param vs baseline (all same params; ours & iso-param both 2x compute & 2x eff-depth).")
