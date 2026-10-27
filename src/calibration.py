"""Fit detection settings and grouping prototypes using local calibration only."""
from pathlib import Path
import numpy as np
import torch
from scipy.signal import find_peaks
from boundary_core import batch_probs,grid_best,score,summary
from baseline_common import normalized_pdw
from bilstm import BiLstmChangePointDetector
from lstm_fcn import LstmFcnWindowClassifier
from cpc_inference import alignment,cpc_scores
import reproduction as r

def calibrate(output,device='cpu'):
    import ruptures as rpt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    values=r.samples('calibration')
    if len(values)<7:raise ValueError('At least seven calibration sequences are required for seven grouping prototypes.')
    for name,method in [('bilstm','BiLSTM'),('lstm_fcn','LSTM-FCN')]:
        md='cpu' if name=='lstm_fcn' else device
        model=BiLstmChangePointDetector() if name=='bilstm' else LstmFcnWindowClassifier()
        model.load_state_dict(torch.load(r.checkpoint_path(name),map_location='cpu',weights_only=True)['state_dict'])
        model=model.to(md).eval()
        winner,grid=grid_best(batch_probs(model,method,values,md),values,method)
        r.write_csv(output/(method+'_calibration_grid.csv'),grid)
        r.write_json(output/(method+'_calibration.json'),dict(threshold=winner['threshold'],prominence=winner['prominence'],min_distance=50,calibration_sequences=len(values)))
        print('CALIBRATE',method,'F1',winner['f1'],flush=True)
    model=r.make_cpc()
    model.load_state_dict(torch.load(r.checkpoint_path('cpc'),map_location='cpu',weights_only=True)['state_dict']);model.eval()
    scores=[cpc_scores(model,s['pdw_data']) for s in values];scale,offset=alignment(model);grid=[]
    for prominence in [.001,.003,.01,.03,.1,.3]:
        rows=[]
        for sample,value in zip(values,scores):
            peaks,_=find_peaks(-value,prominence=prominence,distance=10,plateau_size=(1,5))
            rows.append(score([int(p*scale+offset) for p in peaks],sample,'CPC','calibration',0.))
        grid.append(dict(prominence=prominence,**summary(rows)))
    cpc=max(grid,key=lambda row:row['f1']);r.write_csv(output/'CPC_calibration_grid.csv',grid)
    penalties=[4.,8.,16.,32.,64.];by_penalty={penalty:[] for penalty in penalties}
    for sample in values:
        x=normalized_pdw(sample);algorithm=rpt.Pelt(model='rbf',min_size=20,jump=5).fit(x)
        for penalty in penalties:
            cuts=[int(v) for v in algorithm.predict(pen=penalty) if 0<v<len(x)]
            by_penalty[penalty].append(score(cuts,sample,'PELT','calibration',0.))
    grid=[dict(penalty=penalty,**summary(rows)) for penalty,rows in by_penalty.items()]
    pelt=max(grid,key=lambda row:row['f1']);r.write_csv(output/'PELT_calibration_grid.csv',grid)
    r.write_json(output/'unsupervised_calibration.json',dict(CPC_prominence=cpc['prominence'],PELT_penalty=pelt['penalty'],calibration_sequences=len(values)))
    full,_=r.load_ldcr('full',device)
    r.frozen_references(output,refit=True,model=full,device=device)
    r.write_json(output/'calibration_report.json',dict(source_split='calibration',sequences=len(values),
        data_sha256=r.sha(Path(r.DATA_DIR)/'calibration.npz'),full_checkpoint_sha256=r.sha(r.checkpoint_path('full')),
        checkpoints={name:r.sha(r.checkpoint_path(name)) for name in list(r.LDCR_NAMES)+['cpc','bilstm','lstm_fcn']},
        prototype_clusters=7,test_data_used=False,CPC_prominence=cpc['prominence'],PELT_penalty=pelt['penalty']))
    print('CALIBRATE: all thresholds and three grouping references fitted from local calibration data.',flush=True)
