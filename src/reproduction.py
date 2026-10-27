"""Evaluate freshly trained models using assets from one local experiment."""
from pathlib import Path
from types import SimpleNamespace
import csv,json,hashlib,platform,shutil,time
import numpy as np
import torch
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import adjusted_rand_score
import dataset_generator as gen
import sequence_core as seq
from paired_corruptions import corrupt
from train_infer_ldcr_net import load_checkpoint
from revision_utils import match_boundaries,summarize_counts
from ablation_core import predict as ldcr_predict
from boundary_core import score,summary,probs
from baseline_common import normalized_pdw,probabilities_to_change_points
from bilstm import BiLstmChangePointDetector
from lstm_fcn import LstmFcnWindowClassifier
from CPC import NextFrameClassifier
from cpc_inference import alignment,cpc_scores

PACKAGE=Path(__file__).resolve().parents[1]
CHECKPOINT_DIR=None
CALIBRATION_DIR=None
DATA_DIR=None

def checkpoint_path(name,directory=None):
    folder=directory or CHECKPOINT_DIR
    if folder is None:raise FileNotFoundError('No checkpoint directory configured. Run generate and train first.')
    candidate=Path(folder)/(name+'.pt')
    if not candidate.is_file():raise FileNotFoundError(f'Missing {candidate}. Run train for this experiment first.')
    return candidate

def calibration_path(name):
    if CALIBRATION_DIR is None:raise FileNotFoundError('Run generate, train and calibrate first.')
    candidate=Path(CALIBRATION_DIR)/(name+'_calibration.json')
    if not candidate.is_file():raise FileNotFoundError(f'Missing {candidate}. Run calibrate first.')
    return candidate
PROTOCOL=json.loads((PACKAGE/'config/experiment.json').read_text())
LDCR_NAMES={'full':'LDCR-Net','no_median':'No median','no_multiscale':'No multiscale',
            'no_reconstruction':'No reconstruction','no_tv':'No TV','no_invariance':'No invariance'}
def boundary_conditions():
    return [('random_loss',v) for v in PROTOCOL['main_loss_levels']]+[('receiver_error',v) for v in PROTOCOL['receiver_error_levels']]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:raise ValueError(f'No records to write: {path}')
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def samples(split,limit=None,directory=None):
    folder=directory or DATA_DIR
    if folder is None:raise FileNotFoundError('Run generate first to create local synthetic data.')
    path=Path(folder)/(split+'.npz')
    if not path.is_file():raise FileNotFoundError(f'Missing {path}. Run generate first.')
    with np.load(path,allow_pickle=True) as f:values=list(f['data'])
    return values if limit is None else values[:limit]
def load_ldcr(name,device='cpu',checkpoint_dir=None):
    path=checkpoint_path(name,checkpoint_dir)
    model,state=load_checkpoint(str(path),device)
    return model,state

def make_cpc():
    hp=SimpleNamespace(z_dim=64,latent_dim=128,input_features=4,z_proj=32,n_negatives=10,
                       batch_shuffle=True,strides=[2,2],pool_kernels=[5,5],use_median_pool=True)
    return NextFrameClassifier(hp,rnn=False,embedding=False,kernel_size=[2,2,2,1,1])
def detector(stride):
    return dict(windows=[int(w/stride) for w in [12,16,20]],sigma=2/stride,
                width=2/stride,distance=50/stride,prominence=.30)


def frozen_references(output,refit=False,model=None,device='cpu',existing=False):
    output=Path(output);seq.HERE=output
    if refit:
        rows=[seq.extract(model,s,device) for s in samples('calibration')]
        return seq.fit_references(rows)
    references={}
    for mode in seq.MODES:
        path=output/(mode+'_frozen_reference.npz') if existing else Path(CALIBRATION_DIR)/(mode+'_frozen_reference.npz')
        with np.load(path,allow_pickle=True) as f:
            centers=f['centers'];mapping=f['scoring_mapping'].item()
        estimator=KMeans(n_clusters=7,n_init=20,random_state=42)
        estimator.cluster_centers_=centers;estimator.n_features_in_=centers.shape[1];estimator._n_threads=2
        references[mode]=(estimator,mapping)
        if not existing:shutil.copy2(path,output/(mode+'_frozen_reference.npz'))
    return references

def run_sequence(output,limit=None,device='cpu',extended=True,refit=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);seq.HERE=output
    values=samples('test',limit);model,_=load_ldcr('full',device)
    refs=frozen_references(output,refit,model,device)
    results=[];grouping={};boundaries=[]
    clean=[seq.extract(model,s,device) for s in values]
    np.savez_compressed(output/'test_clean_detected_features.npz',data=np.array(clean,dtype=object))
    losses=PROTOCOL['extended_loss_levels'] if extended else PROTOCOL['main_loss_levels']
    for family,levels in [('random_loss',losses),('receiver_error',PROTOCOL['receiver_error_levels'])]:
        for level in levels:
            if level==0:rows=clean
            else:
                rows=[seq.extract(model,corrupt(s,family,level,4100000+int(s['sample_id'])),device) for s in values]
                tag=family+'_'+str(float(level)).replace('.','_')+'_detected_features.npz'
                np.savez_compressed(output/tag,data=np.array(rows,dtype=object))
            rr,ari=seq.score_rows(rows,refs,family,level);results.extend(rr);grouping[family,level]=ari
            counts=[match_boundaries(r['detected_cuts'],r['true_cuts'],20)[0] for r in rows]
            boundaries.append({'family':family,'level':level,**summarize_counts(counts)})
            print('SEQUENCE',family,level,'N',len(rows),'ARI',ari,flush=True)
    seq.summarize(results,grouping)
    write_csv(output/'paired_boundary_summary.csv',boundaries)
    write_json(output/'completed_run.json',dict(completed=True,test_sequences=len(values),conditions=len(losses)+len(PROTOCOL['receiver_error_levels']),
        extended=extended,refit_calibration=refit,checkpoint_sha256=sha(checkpoint_path('full'))))
    return results

def bootstrap_global(output,replicates=1000):
    """Same RNG order and whole-sequence resampling as the published routine."""
    output=Path(output);records=read_csv(output/'paired_summary.csv')
    refs=frozen_references(output,existing=True);intervals=[];differences=[];rng=np.random.default_rng(860170)
    cases=list(dict.fromkeys((r['family'],float(r['level'])) for r in records))
    for family,level in cases:
        tag='test_clean_detected' if level==0 else family+'_'+str(level).replace('.','_')+'_detected'
        with np.load(output/(tag+'_features.npz'),allow_pickle=True) as f:rows=list(f['data'])
        truth=[r['semantics'] for r in rows]
        states={mode:[refs[mode][0].predict(normalize(r['features'][mode]).astype(refs[mode][0].cluster_centers_.dtype)) for r in rows] for mode in refs}
        estimates={mode:float(adjusted_rand_score(np.concatenate(truth),np.concatenate(v))) for mode,v in states.items()}
        n=len(rows);indices=rng.integers(0,n,(replicates,n));boots={mode:[] for mode in refs}
        for selected in indices:
            target=np.concatenate([truth[i] for i in selected])
            for mode in refs:boots[mode].append(adjusted_rand_score(target,np.concatenate([states[mode][i] for i in selected])))
        for mode in refs:
            lo,hi=np.quantile(boots[mode],[.025,.975])
            intervals.append(dict(family=family,level=level,method=mode,ari_global=estimates[mode],ci_low=float(lo),ci_high=float(hi),bootstrap_sequences=n,bootstrap_replicates=replicates))
        for other in ['ti','pdw']:
            delta=np.array(boots['tv'])-np.array(boots[other]);lo,hi=np.quantile(delta,[.025,.975])
            differences.append(dict(family=family,level=level,contrast='tv - '+other,metric='ari_global',difference=estimates['tv']-estimates[other],ci_low=float(lo),ci_high=float(hi)))
        print('BOOTSTRAP',family,level,flush=True)
    write_csv(output/'global_ari_bootstrap.csv',intervals);write_csv(output/'global_ari_paired_differences.csv',differences)

def run_boundary(output,limit=None,device='cpu',methods=None):
    import ruptures as rpt
    output=Path(output);output.mkdir(parents=True,exist_ok=True);values=samples('test',limit)
    names=list(LDCR_NAMES)+['cpc','bilstm','lstm_fcn','pelt'] if methods is None else methods
    models={}
    for name in names:
        if name in LDCR_NAMES:models[name]=load_ldcr(name,device)[0]
        elif name in ['bilstm','lstm_fcn']:
            m=BiLstmChangePointDetector() if name=='bilstm' else LstmFcnWindowClassifier()
            md='cpu' if name=='lstm_fcn' else device
            m.load_state_dict(torch.load(checkpoint_path(name),map_location=md,weights_only=False)['state_dict'])
            models[name]=m.to(md).eval()
        elif name=='cpc':
            m=make_cpc()
            m.load_state_dict(torch.load(checkpoint_path('cpc'),map_location='cpu',weights_only=True)['state_dict']);models[name]=m.eval()
    conf={name:json.loads(calibration_path(name).read_text()) for name in ['BiLSTM','LSTM-FCN','unsupervised']}
    records=[];stats=[]
    for family,level in boundary_conditions():
        observed=[corrupt(s,family,level,4100000+int(s['sample_id'])) for s in values]
        for name in names:
            label=LDCR_NAMES.get(name,{'cpc':'CPC','bilstm':'BiLSTM','lstm_fcn':'LSTM-FCN','pelt':'PELT'}.get(name))
            rows=[]
            for s in observed:
                if name in LDCR_NAMES:
                    model=models[name]
                    # The matched-stride detector routine receives CPU tensors.
                    if device!='cpu':model=model.cpu()
                    pred=ldcr_predict(model,s,detector(model.output_stride))
                elif name=='pelt':
                    x=normalized_pdw(s);pred=[int(v) for v in rpt.Pelt(model='rbf',min_size=20,jump=5).fit(x).predict(pen=conf['unsupervised']['PELT_penalty']) if 0<v<len(x)]
                elif name=='cpc':
                    v=cpc_scores(models[name],s['pdw_data']);peaks,_=find_peaks(-v,prominence=conf['unsupervised']['CPC_prominence'],distance=10,plateau_size=(1,5));scale,offset=alignment(models[name]);pred=[int(p*scale+offset) for p in peaks]
                else:
                    c=conf[label];md='cpu' if name=='lstm_fcn' else device
                    p=probs(models[name],label,s,md);pred=probabilities_to_change_points(p,c['threshold'],50,c['prominence'])
                rows.append(score(pred,s,label,family,level))
            records.extend(rows);stats.append(summary(rows))
            print('BOUNDARY',label,family,level,stats[-1]['f1'],flush=True)
            write_csv(output/'boundary_records.csv',records)
    write_csv(output/'all_boundary_summary.csv',stats)
    boundary=[r for r in stats if r['method'] in ['LDCR-Net','CPC','PELT','BiLSTM','LSTM-FCN']]
    ablation=[r for r in stats if r['method'] in LDCR_NAMES.values()]
    if boundary:write_csv(output/'boundary_summary.csv',boundary)
    if ablation:write_csv(output/'ablation_summary.csv',ablation)
    return records

def reference_consistency(output,limit=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);seq.HERE=output
    model,_=load_ldcr('full');refs=frozen_references(output,existing=all((output/(m+'_frozen_reference.npz')).exists() for m in seq.MODES));center=refs['tv'][0].cluster_centers_;records=[]
    def assign(r):
        x=normalize(r['features']['tv']).astype(center.dtype)
        return np.square(x[:,None,:]-center[None,:,:]).sum(-1).argmin(1)
    for s in samples('test',limit):
        baseline=seq.extract(model,s,'cpu');radar=s['radar_pdw'].copy();jammer=s['jammer_pdw'].copy()
        nominal=gen.compute_relative_pdw(radar,jammer);noise=s['pdw_data']-nominal;half=len(radar)//2
        for x in [radar,jammer]:x[half:,1]+=500.;x[half:,2]*=1.5;x[half:,3]+=6.
        relative=gen.compute_relative_pdw(radar,jammer);changed=dict(s,radar_pdw=radar,jammer_pdw=jammer,pdw_data=relative+noise)
        result=seq.extract(model,changed,'cpu');diff=np.abs(relative-nominal).max(0)
        b=np.array_equal(baseline['boundaries'],result['boundaries']);ids=np.array_equal(assign(baseline),assign(result))
        records.append(dict(sample=int(s['sample_id']),num_events=len(radar),max_delay_difference=float(diff[0]),max_frequency_difference=float(diff[1]),max_width_ratio_difference=float(diff[2]),max_amplitude_difference=float(diff[3]),max_rpdw_difference=float(diff.max()),identical_float32_input=int(np.array_equal(s['pdw_data'].astype(np.float32),changed['pdw_data'].astype(np.float32))),identical_boundaries=int(b),identical_segment_identities=int(ids),identical_recovered_intervals=int(b and ids)))
    write_csv(output/'reference_consistency.csv',records)
    result=dict(sequences=len(records),max_rpdw_difference=max(r['max_rpdw_difference'] for r in records),identical_boundary_count=sum(r['identical_boundaries'] for r in records),identical_identity_count=sum(r['identical_segment_identities'] for r in records),identical_interval_sequence_count=sum(r['identical_recovered_intervals'] for r in records),identical_float32_input_count=sum(r['identical_float32_input'] for r in records))
    write_json(output/'reference_consistency.json',result);print('REFERENCE',result,flush=True)

def complexity(output):
    model,_=load_ldcr('full');groups=['multiscale','cr_block','ti_head','tv_head','reconstruction_head'];counts={g:0 for g in groups};flops={g:0 for g in groups};handles=[]
    for name,p in model.named_parameters():counts[name.split('.')[0]]+=p.numel()
    def hook(name):
        def add(module,args,out):flops[name.split('.')[0]]+=2*out.numel()*(module.in_channels//module.groups)*module.kernel_size[0]
        return add
    for name,m in model.named_modules():
        if isinstance(m,torch.nn.Conv1d):handles.append(m.register_forward_hook(hook(name)))
    with torch.inference_mode():model(torch.zeros(1,4,1000))
    for h in handles:h.remove()
    rows=[dict(module=g,parameters=counts[g],parameters_million=counts[g]/1e6,gflops=flops[g]/1e9) for g in groups]
    for label,selected in [('inference_encoder',groups[:-1]),('full_model',groups)]:rows.append(dict(module=label,parameters=sum(counts[g] for g in selected),parameters_million=sum(counts[g] for g in selected)/1e6,gflops=sum(flops[g] for g in selected)/1e9))
    write_csv(Path(output)/'complexity.csv',rows);print('COMPLEXITY',rows[-2:],flush=True)

