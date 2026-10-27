#!/usr/bin/env python
"""Generate synthetic data, train, calibrate, evaluate and plot a new experiment."""
from pathlib import Path
import argparse,os,sys

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
os.environ.setdefault('MPLBACKEND','Agg')
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(name,'2')
sys.path.insert(0,str(ROOT/'src'))

def positive(value):
    value=int(value)
    if value<1:raise argparse.ArgumentTypeError('must be a positive integer')
    return value

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ['pipeline','generate','train','calibrate','evaluate','plot','check']:
        q=commands.add_parser(name)
        q.add_argument('--workdir',type=Path,default=ROOT/'runs/full',help='All locally generated experiment assets go here')
        if name in ['pipeline','generate']:
            q.add_argument('--profile',choices=['smoke','full'],default='full')
            q.add_argument('--seed',type=int,help='Override the base data-generation seed; the three splits use consecutive seeds')
            for split in ['training','calibration','test']:
                q.add_argument('--'+split+'-size',type=positive)
        if name in ['pipeline','train','calibrate','evaluate']:
            q.add_argument('--device',choices=['cpu','cuda'],default='cpu')
        if name in ['pipeline','train']:
            q.add_argument('--epochs',type=positive,help='Override all selected models epoch limits')
        if name=='train':
            q.add_argument('--model',default='all',choices=['all','full','no_median','no_multiscale','no_reconstruction','no_tv','no_invariance','cpc','bilstm','lstm_fcn'])
    args=parser.parse_args();args.workdir=args.workdir.resolve()
    os.environ.setdefault('MPLCONFIGDIR',str(args.workdir/'.mplconfig'))
    import torch
    from threadpoolctl import threadpool_limits
    from workflow import dispatch
    torch.set_num_threads(2)
    try:
        with threadpool_limits(limits=2):dispatch(args)
    except (FileNotFoundError,ValueError) as error:
        parser.exit(2,f'Error: {error}\n')

if __name__=='__main__':main()
