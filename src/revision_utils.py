import numpy as np


def match_boundaries(predicted, truth, tolerance):
    """Maximum-cardinality order-preserving match, then minimum absolute error."""
    p, t = sorted(map(int,predicted)), sorted(map(int,truth))
    n,m=len(p),len(t)
    dp=[[ (0,0,[]) for _ in range(m+1)] for _ in range(n+1)]
    for i in range(1,n+1):
        for j in range(1,m+1):
            choices=[dp[i-1][j],dp[i][j-1]]
            dist=abs(p[i-1]-t[j-1])
            if dist<=tolerance:
                c,e,pairs=dp[i-1][j-1]
                choices.append((c+1,e+dist,pairs+[(p[i-1],t[j-1])]))
            dp[i][j]=max(choices,key=lambda v:(v[0],-v[1]))
    c,_,pairs=dp[n][m]
    return (c,n-c,m-c),pairs


def summarize_counts(counts,seed=20260928,repeats=1000):
    counts=np.asarray(counts,dtype=int)
    tp,fp,fn=counts.sum(axis=0)
    f1=lambda a:2*a[...,0]/np.maximum(2*a[...,0]+a[...,1]+a[...,2],1)
    rng=np.random.default_rng(seed)
    boot=counts[rng.integers(0,len(counts),size=(repeats,len(counts)))].sum(axis=1)
    lo,hi=np.quantile(f1(boot),[.025,.975])
    return {'num_sequences':len(counts),'tp':int(tp),'fp':int(fp),'fn':int(fn),
            'precision':float(tp/max(tp+fp,1)), 'recall':float(tp/max(tp+fn,1)),
            'f1':float(f1(np.array([tp,fp,fn]))),'ci_low':float(lo),'ci_high':float(hi),
            'false_alarms_per_sequence':float(fp/len(counts))}
