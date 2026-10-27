import numpy as np
import torch
from torch.utils.data import Dataset
from revision_utils import match_boundaries,summarize_counts
from baseline_common import normalized_pdw,change_points_to_labels,probabilities_to_change_points
from lstm_fcn import infer_sequence_probabilities


CONDITIONS=[('random_loss',v) for v in [0.,.05,.10,.15,.20]]+[('receiver_error',v) for v in [0.,.5,1.,1.5,2.]]


def score(pred,s,method,family,level):
    (tp,fp,fn),pairs=match_boundaries(pred,s['change_points'],20)
    return {'method':method,'family':family,'level':level,'sample':int(s.get('sample_id',0)),'n':len(s['pdw_data']),'tp':tp,'fp':fp,'fn':fn,'error_sum':sum(abs(a-b) for a,b in pairs),'truth':';'.join(map(str,s['change_points'])),'predicted':';'.join(map(str,pred))}


def summary(rows):
    r=summarize_counts([[v[k] for k in ['tp','fp','fn']] for v in rows],seed=7031,repeats=2000)
    return {'method':rows[0]['method'],'family':rows[0]['family'],'level':rows[0]['level'],**r,'matched_mae':sum(v['error_sum'] for v in rows)/max(r['tp'],1)}


class Windows(Dataset):
    def __init__(self,samples):
        self.x=[normalized_pdw(s) for s in samples];self.index=[];rng=np.random.default_rng(42)
        for i,s in enumerate(samples):
            cps=np.asarray(s['change_points']);centers=np.unique(np.r_[np.arange(0,len(self.x[i]),4),np.concatenate([np.arange(max(0,int(cp)-6),min(len(self.x[i]),int(cp)+7)) for cp in cps])])
            if len(centers)>80:centers=np.sort(rng.choice(centers,80,replace=False))
            self.index.extend((i,int(c),float(np.any(abs(cps-c)<=6))) for c in centers)
    def __len__(self):return len(self.index)
    def __getitem__(self,i):
        j,c,y=self.index[i];x=self.x[j];w=np.zeros((96,4),np.float32);a=c-48;b=a+96;lo=max(0,a);hi=min(len(x),b);w[lo-a:hi-a]=x[lo:hi]
        return torch.from_numpy(w),torch.tensor(y,dtype=torch.float32)


class Chunks(Dataset):
    """Equal-length labeled crops permit fused recurrent kernels without padding."""
    def __init__(self,samples):
        self.x=[normalized_pdw(s) for s in samples]
        self.y=[change_points_to_labels(len(x),s['change_points'],3) for x,s in zip(self.x,samples)]
        self.index=[]
        for i,x in enumerate(self.x):
            assert len(x)>=256
            self.index.extend((i,int(a)) for a in np.unique(np.r_[np.arange(0,len(x)-255,128),len(x)-256]))
    def __len__(self):return len(self.index)
    def __getitem__(self,i):
        j,a=self.index[i];return torch.from_numpy(self.x[j][a:a+256]),torch.from_numpy(self.y[j][a:a+256])


def probs(model,method,s,device):
    if method=='BiLSTM':
        x=torch.from_numpy(normalized_pdw(s))[None].to(device)
        with torch.inference_mode():return torch.sigmoid(model.head(model.lstm(x)[0]).squeeze(-1)).cpu().numpy()[0]
    return infer_sequence_probabilities(model,s,device,96,2)


def batch_probs(model,method,samples,device):
    return [probs(model,method,s,device) for s in samples]


def grid_best(scores,samples,method):
    grid=[]
    for threshold in [.2,.35,.5,.65,.8]:
        for prominence in [.01,.05,.1]:
            rows=[score(probabilities_to_change_points(p,threshold,50,prominence),s,method,'calibration',0.) for p,s in zip(scores,samples)]
            r=summary(rows);grid.append({**r,'threshold':threshold,'prominence':prominence})
    best=max(grid,key=lambda r:r['f1']);return best,grid
