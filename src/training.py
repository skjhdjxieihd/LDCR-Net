"""Fresh training on locally generated data for all learned comparison models."""
from pathlib import Path
from dataclasses import asdict
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from train_infer_ldcr_net import (PdwDataset,collate_batch,ModelConfig,LossConfig,
    make_model,training_loss,seed_everything)
from boundary_core import Chunks,Windows,batch_probs,grid_best
from baseline_common import change_points_to_labels
from bilstm import BiLstmChangePointDetector
from lstm_fcn import LstmFcnWindowClassifier
from reproduction import samples,write_csv,write_json,sha,make_cpc

def metadata(args,epochs,count):
    return dict(model=args.model,seed=args.config['seed'],epochs=epochs,training_sequences=count,
                training_data_sha256=sha(args.training_data),source='locally generated synthetic data')

def train(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    seed_everything(args.config['seed'])
    if args.model in ['bilstm','lstm_fcn']:return train_supervised(args)
    if args.model=='cpc':return train_cpc(args)
    settings=args.config['models'][args.model]
    model_config=ModelConfig(**settings['model_config']);loss_config=LossConfig(**settings['loss_config'])
    ds=PdwDataset(str(args.training_data))
    dl=DataLoader(ds,batch_size=args.batch_size,shuffle=True,collate_fn=collate_batch,num_workers=0)
    model=make_model(model_config).to(args.device);recipe=args.config['training']
    opt=torch.optim.AdamW(model.parameters(),lr=recipe['ldcr_learning_rate'],weight_decay=recipe['weight_decay'])
    history=[];start=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train();totals={}
        for batch in dl:
            opt.zero_grad(set_to_none=True)
            loss,terms=training_loss(model,batch['pdw'].to(args.device),batch['lengths'].to(args.device),loss_config)
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite {args.model} training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),recipe['gradient_clip']);opt.step()
            for key,value in dict(loss=float(loss.detach()),**terms).items():totals[key]=totals.get(key,0.)+value
        history.append(dict(epoch=epoch,**{key:value/len(dl) for key,value in totals.items()},elapsed_seconds=time.perf_counter()-start))
        write_csv(output/(args.model+'_training_history.csv'),history)
        print('TRAIN',args.model,epoch,'loss',history[-1]['loss'],flush=True)
    record=metadata(args,args.epochs,len(ds));target=output/(args.model+'.pt')
    torch.save(dict(state_dict=model.state_dict(),model_config=asdict(model_config),loss_config=asdict(loss_config),
                    optimizer_state_dict=opt.state_dict(),**record),target)
    write_json(output/(args.model+'_training_metadata.json'),dict(**record,checkpoint_sha256=sha(target)))

class CpcCrops(Dataset):
    """Fixed-length raw-RPDW crops avoid padding in the CPC objective."""
    def __init__(self,values,length=256,stride=128):
        self.values=values;self.length=length;self.index=[]
        for i,sample in enumerate(values):
            n=len(sample['pdw_data'])
            if n<length:raise ValueError(f'CPC requires at least {length} events per training sequence')
            for start in np.unique(np.r_[np.arange(0,n-length+1,stride),n-length]):self.index.append((i,int(start)))
    def __len__(self):return len(self.index)
    def __getitem__(self,index):
        i,a=self.index[index]
        return torch.from_numpy(np.asarray(self.values[i]['pdw_data'][a:a+self.length],np.float32).T.copy())

def train_cpc(args):
    recipe=args.config['training'];values=samples('training',directory=args.data_dir)
    ds=CpcCrops(values,recipe['cpc_crop_length'],recipe['cpc_crop_stride'])
    dl=DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=0)
    model=make_cpc().to(args.device);opt=torch.optim.Adam(model.parameters(),lr=recipe['cpc_learning_rate'])
    history=[];start=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train();losses=[]
        for x in dl:
            opt.zero_grad(set_to_none=True);loss=model.loss(model(x.to(args.device)))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite CPC training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),elapsed_seconds=time.perf_counter()-start))
        write_csv(Path(args.output)/'cpc_training_history.csv',history);print('TRAIN cpc',history[-1],flush=True)
    record=metadata(args,args.epochs,len(values));target=Path(args.output)/'cpc.pt'
    torch.save(dict(state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),**record),target)
    write_json(Path(args.output)/'cpc_training_metadata.json',dict(**record,checkpoint_sha256=sha(target),crop_length=ds.length))

def train_supervised(args):
    method='BiLSTM' if args.model=='bilstm' else 'LSTM-FCN'
    values=samples('training',directory=args.data_dir)[:args.config['training']['supervised_training_limit']]
    calibration=samples('calibration',directory=args.data_dir);validation=calibration[:100]
    if method=='BiLSTM':
        model=BiLstmChangePointDetector().to(args.device);ds=Chunks(values)
        positives=sum(change_points_to_labels(len(s['pdw_data']),s['change_points'],3).sum() for s in values)
        total=sum(len(s['pdw_data']) for s in values);weight=(total-positives)/max(positives,1)
        batch_size=64 if args.config['profile']=='full' else args.batch_size
    else:
        model=LstmFcnWindowClassifier().to(args.device);ds=Windows(values)
        positives=sum(y for _,_,y in ds.index);weight=(len(ds)-positives)/max(positives,1)
        batch_size=128 if args.config['profile']=='full' else args.batch_size
    dl=DataLoader(ds,batch_size=batch_size,shuffle=True,num_workers=0)
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight],device=args.device),reduction='none')
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    best=-1.;significant=-1.;last=0;history=[];start=time.perf_counter();target=Path(args.output)/(args.model+'.pt')
    for epoch in range(1,args.epochs+1):
        model.train();losses=[]
        for x,y in dl:
            opt.zero_grad(set_to_none=True)
            prediction=model.head(model.lstm(x.to(args.device))[0]).squeeze(-1) if method=='BiLSTM' else model(x.to(args.device))
            loss=criterion(prediction,y.to(args.device)).mean()
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite {method} training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();losses.append(float(loss.detach()))
        model.eval();winner,_=grid_best(batch_probs(model,method,validation,args.device),validation,method)
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),validation_f1=winner['f1'],elapsed_seconds=time.perf_counter()-start))
        write_csv(Path(args.output)/(args.model+'_training_history.csv'),history);print('TRAIN',method,history[-1],flush=True)
        if winner['f1']>best:
            best=winner['f1']
            torch.save(dict(state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),architecture=method,
                            label_radius=3 if method=='BiLSTM' else 6,**metadata(args,epoch,len(values))),target)
        if winner['f1']>significant+.001:significant=winner['f1'];last=epoch
        if winner['f1']==1. or epoch-last>=3:break
    chosen=torch.load(target,map_location='cpu',weights_only=True)
    write_json(Path(args.output)/(args.model+'_training_metadata.json'),dict(**metadata(args,chosen['epochs'],len(values)),checkpoint_sha256=sha(target),selected_validation_f1=best))
