"""Run-local generation, training, calibration, evaluation and integrity checks."""
from pathlib import Path
from types import SimpleNamespace
import json,platform,time
import numpy as np
import torch
import reproduction as r

MODELS=list(r.LDCR_NAMES)+['cpc','bilstm','lstm_fcn']

def validate_recipe(config):
    main=config['main_loss_levels'];extended=config['extended_loss_levels'];errors=config['receiver_error_levels']
    if (len(main)!=len(set(main)) or len(extended)!=len(set(extended)) or len(errors)!=len(set(errors))
        or not {0.,.2}.issubset(main) or not set(main).issubset(extended)
        or not {0.,2.}.issubset(errors) or not all(0<=v<=.2 for v in main)
        or not all(0<=v<=.6 for v in extended) or max(extended)<=.2 or not all(0<=v<=2. for v in errors)):
        raise ValueError('Figure settings require unique main loss levels within 0–0.2 including 0/0.2; extended loss must include the main levels and reach above 0.2 within 0–0.6; receiver errors must lie within 0–2 including 0/2.')

def read_config(workdir):
    path=Path(workdir)/'run_config.json'
    if not path.is_file():raise FileNotFoundError(f'Missing {path}. Run generate --workdir "{workdir}" first.')
    config=json.loads(path.read_text(encoding='utf-8'))
    validate_recipe(config)
    r.DATA_DIR=Path(workdir)/'data';r.CHECKPOINT_DIR=Path(workdir)/'checkpoints';r.CALIBRATION_DIR=Path(workdir)/'calibration'
    r.PROTOCOL=config
    return config

def generate_corpora(args):
    workdir=Path(args.workdir)
    if workdir.exists() and any(workdir.iterdir()):
        raise ValueError('Generation requires an empty workdir. Choose a new directory to preserve previous experiments.')
    config=json.loads((r.PACKAGE/'config/experiment.json').read_text(encoding='utf-8'))
    validate_recipe(config)
    config['profile']=args.profile;config['active']=dict(config['profiles'][args.profile])
    if args.seed is not None:config['generation']['seeds']={name:args.seed+i for i,name in enumerate(['training','calibration','test'])}
    for split in ['training','calibration','test']:
        count=getattr(args,split+'_size',None)
        if count is not None:config['active'][split+'_size']=count
    if config['active']['calibration_size']<7:raise ValueError('calibration-size must be at least 7 for seven grouping prototypes.')
    directory=workdir/'data';directory.mkdir(parents=True)
    generation={}
    for split in ['training','calibration','test']:
        seed=config['generation']['seeds'][split];count=config['active'][split+'_size']
        values=r.gen.generate_jamming_pdw_dataset(count,None,seed=seed,perturb_config={'loss_ratio':0.},
            num_state_changes_range=tuple(config['generation']['state_changes']),
            num_pulses_per_segment_range=tuple(config['generation']['events_per_segment']))
        for i,sample in enumerate(values):sample.update(sample_id=i,generation_seed=seed,split=split)
        path=directory/(split+'.npz');np.savez_compressed(path,data=np.array(values,dtype=object))
        generation[split]=dict(seed=seed,sequences=count,sha256=r.sha(path))
        print('GENERATE',split,count,'seed',seed,flush=True)
    r.write_json(workdir/'run_config.json',config)
    r.write_json(workdir/'generation.json',dict(source='new local synthetic generation',splits=generation))
    read_config(workdir)

def train_models(workdir,device='cpu',epochs=None,model='all'):
    from training import train
    config=read_config(workdir);settings=config['active']
    names=MODELS if model=='all' else [model]
    for name in names:
        category='supervised_epochs' if name in ['bilstm','lstm_fcn'] else 'cpc_epochs' if name=='cpc' else 'invariance_epochs' if name=='no_invariance' else 'ldcr_epochs'
        args=SimpleNamespace(model=name,config=config,output=Path(workdir)/'checkpoints',
            training_data=Path(workdir)/'data/training.npz',data_dir=Path(workdir)/'data',device=device,
            epochs=epochs or settings[category],batch_size=settings['batch_size'])
        train(args)

def calibrate_models(workdir,device='cpu'):
    from calibration import calibrate
    read_config(workdir)
    for name in MODELS:r.checkpoint_path(name)
    calibrate(Path(workdir)/'calibration',device)

def validate_calibration_assets(workdir):
    path=Path(workdir)/'calibration/calibration_report.json'
    if not path.is_file():raise FileNotFoundError('Run calibrate before evaluate.')
    report=json.loads(path.read_text(encoding='utf-8'))
    current={name:r.sha(r.checkpoint_path(name)) for name in MODELS}
    if (report.get('checkpoints')!=current or report.get('full_checkpoint_sha256')!=current['full']
        or report.get('data_sha256')!=r.sha(Path(workdir)/'data/calibration.npz')):
        raise ValueError('Calibration assets are stale. Run calibrate and evaluate again after changing models or data.')
    return report

def validate_evaluation_assets(workdir):
    validate_calibration_assets(workdir)
    path=Path(workdir)/'results/evaluation_metadata.json'
    if not path.is_file():raise FileNotFoundError('Run evaluate before plot or check.')
    report=json.loads(path.read_text(encoding='utf-8'))
    current={name:r.sha(r.checkpoint_path(name)) for name in MODELS}
    if report.get('checkpoints')!=current or report.get('test_data_sha256')!=r.sha(Path(workdir)/'data/test.npz'):
        raise ValueError('Evaluation assets are stale. Run evaluate again after changing models or test data.')
    return report

def evaluate(workdir,device='cpu'):
    config=read_config(workdir);output=Path(workdir)/'results';output.mkdir(exist_ok=True)
    for name in MODELS:r.checkpoint_path(name)
    validate_calibration_assets(workdir)
    r.run_sequence(output,device=device,extended=True)
    r.run_boundary(output,device=device)
    r.reference_consistency(output);r.complexity(output)
    r.bootstrap_global(output,config['active']['bootstrap_replicates'])
    r.write_json(output/'evaluation_metadata.json',dict(source='newly trained models and locally generated test data',
        test_data_sha256=r.sha(Path(workdir)/'data/test.npz'),
        checkpoints={name:r.sha(r.checkpoint_path(name)) for name in MODELS},test_sequences=config['active']['test_size']))

def plot(workdir):
    from plotting import plot_all
    read_config(workdir)
    validate_evaluation_assets(workdir)
    plot_all(Path(workdir)/'figures',Path(workdir)/'results')
    from plotting import plot_training
    plot_training(Path(workdir)/'checkpoints',Path(workdir)/'figures')
    workdir=Path(workdir)
    inputs=[workdir/'results'/name for name in ['boundary_summary.csv','ablation_summary.csv','paired_summary.csv',
             'global_ari_bootstrap.csv','random_loss_0_2_detected_features.npz','tv_frozen_reference.npz','evaluation_metadata.json']]
    inputs.extend(workdir/'checkpoints'/(name+'_training_history.csv') for name in MODELS)
    r.write_json(workdir/'plot_metadata.json',dict(
        inputs={p.relative_to(workdir).as_posix():r.sha(p) for p in inputs},
        outputs={p.relative_to(workdir).as_posix():r.sha(p) for p in sorted((workdir/'figures').iterdir()) if p.is_file()}))

def validate_plot_assets(workdir):
    workdir=Path(workdir);path=workdir/'plot_metadata.json'
    if not path.is_file():raise FileNotFoundError('Run plot before check.')
    record=json.loads(path.read_text(encoding='utf-8'))
    for group in ['inputs','outputs']:
        for name,expected in record[group].items():
            if not (workdir/name).is_file() or r.sha(workdir/name)!=expected:
                raise ValueError(f'Plot assets are stale: {name}. Run plot again after evaluation.')

def check(workdir):
    config=read_config(workdir);workdir=Path(workdir);count=config['active']['test_size'];checks=[]
    generation=json.loads((workdir/'generation.json').read_text())
    seeds=[v['seed'] for v in generation['splits'].values()]
    if len(set(seeds))!=3:raise ValueError('Training, calibration and test generation seeds must be distinct.')
    for split,record in generation['splits'].items():
        if r.sha(workdir/'data'/(split+'.npz'))!=record['sha256']:raise ValueError(f'{split} data changed after generation.')
        if len(r.samples(split))!=record['sequences']:raise ValueError(f'{split} sample count mismatch.')
    checks.append('Three local corpora have distinct seeds, consistent counts and unchanged file hashes.')
    training_hash=r.sha(workdir/'data/training.npz')
    for name in MODELS:
        meta=json.loads((workdir/'checkpoints'/(name+'_training_metadata.json')).read_text())
        if meta['training_data_sha256']!=training_hash or meta['epochs']<1:raise ValueError(f'{name} lacks valid fresh-training provenance.')
        if meta['checkpoint_sha256']!=r.sha(r.checkpoint_path(name)):raise ValueError(f'{name} checkpoint changed after training.')
        rows=r.read_csv(workdir/'checkpoints'/(name+'_training_history.csv'))
        if not rows or not all(np.isfinite(float(row['loss'])) for row in rows):raise ValueError(f'{name} has invalid training loss.')
    checks.append('All nine checkpoints trace to this generated training corpus and finite training histories.')
    calibration=validate_calibration_assets(workdir)
    if calibration['test_data_used'] or calibration['data_sha256']!=generation['splits']['calibration']['sha256']:raise ValueError('Calibration provenance mismatch.')
    checks.append('Detection settings and three prototype sets were fitted from calibration only.')
    validate_evaluation_assets(workdir)
    checks.append('Calibration and evaluation fingerprints match the current models and data.')
    results=workdir/'results'
    boundary_groups=len(r.boundary_conditions());sequence_groups=len(config['extended_loss_levels'])+len(config['receiver_error_levels'])
    expected={'boundary_records.csv':(len(MODELS)+1)*boundary_groups*count,'boundary_summary.csv':5*boundary_groups,'ablation_summary.csv':len(r.LDCR_NAMES)*boundary_groups,
              'paired_per_sequence.csv':3*sequence_groups*count,'paired_summary.csv':3*sequence_groups,'global_ari_bootstrap.csv':3*sequence_groups,
              'reference_consistency.csv':count}
    for name,n in expected.items():
        rows=r.read_csv(results/name)
        if len(rows)!=n:raise ValueError(f'{name}: expected {n} rows, found {len(rows)}.')
        for row in rows:
            for key,value in row.items():
                if key in ['truth','predicted','method','family','contrast']:continue
                try:number=float(value)
                except ValueError:continue
                if not np.isfinite(number):raise ValueError(f'Nonfinite {key} in {name}')
    checks.append('Complete boundary, ablation, sequence, bootstrap and reference results have expected sizes and finite metrics.')
    summaries=r.read_csv(results/'all_boundary_summary.csv');records=r.read_csv(results/'boundary_records.csv')
    for row in summaries:
        subset=[v for v in records if (v['method'],v['family'],float(v['level']))==(row['method'],row['family'],float(row['level']))]
        for key in ['tp','fp','fn']:
            if sum(int(v[key]) for v in subset)!=int(row[key]):raise ValueError(f'Boundary count mismatch for {row["method"]}')
    checks.append('Boundary summaries agree with the underlying per-sequence match counts.')
    validate_plot_assets(workdir)
    names=['figure24_unified_boundary.pdf','figure25_unified_ablation_loss.pdf','figure26_unified_ablation_error.pdf',
           'figure21_paired_pressure.pdf','figure22_paired_timeline.pdf','figure23_extended_loss.pdf','training_losses.pdf']
    for name in names:
        if not (workdir/'figures'/name).read_bytes().startswith(b'%PDF'):raise ValueError(f'Invalid figure: {name}')
    checks.append('Six empirical figures plus the fresh-training loss figure were exported as PDF.')
    report=dict(passed=True,profile=config['profile'],checks=checks,models_trained=len(MODELS),test_sequences=count,
        boundary_predictions=(len(MODELS)+1)*boundary_groups*count,sequence_records=3*sequence_groups*count,bootstrap_replicates=config['active']['bootstrap_replicates'],
        note='Checks establish a runnable new experiment; matching paper numerical results is not required.',
        environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__))
    r.write_json(workdir/'integrity_report.json',report)
    print('CHECK PASSED:',len(checks),'workflow checks.',flush=True)
    return report

def dispatch(args):
    if getattr(args,'device','cpu')=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA is unavailable; use --device cpu or install a compatible PyTorch build.')
    if args.command=='generate':return generate_corpora(args)
    if args.command=='train':return train_models(args.workdir,args.device,args.epochs,args.model)
    if args.command=='calibrate':return calibrate_models(args.workdir,args.device)
    if args.command=='evaluate':return evaluate(args.workdir,args.device)
    if args.command=='plot':return plot(args.workdir)
    if args.command=='check':return check(args.workdir)
    start=time.perf_counter();generate_corpora(args)
    train_models(args.workdir,args.device,args.epochs)
    calibrate_models(args.workdir,args.device);evaluate(args.workdir,args.device);plot(args.workdir)
    report=check(args.workdir);report['elapsed_seconds']=time.perf_counter()-start
    r.write_json(args.workdir/'integrity_report.json',report)
    print('PIPELINE COMPLETE:',args.workdir,flush=True)
