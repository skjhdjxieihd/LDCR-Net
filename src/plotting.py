"""Plot measured results and training losses from a newly executed experiment."""
from pathlib import Path
import tempfile,shutil,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.preprocessing import normalize
from reproduction import read_csv,write_csv,complexity
import plot_sequence,plot_boundary,plot_ablation

def timeline_records(results):
    results=Path(results)
    if (results/'timeline.csv').exists():return read_csv(results/'timeline.csv')
    with np.load(results/'random_loss_0_2_detected_features.npz',allow_pickle=True) as f:rows=list(f['data'])
    row=next(r for r in rows if r['id']==0)
    with np.load(results/'tv_frozen_reference.npz',allow_pickle=True) as f:centers=f['centers'];mapping=f['scoring_mapping'].item()
    x=normalize(row['features']['tv']).astype(centers.dtype)
    assigned=np.argmin(((x[:,None,:]-centers[None,:,:])**2).sum(2),axis=1)
    mapped=np.array([mapping[int(v)] for v in assigned])
    pred=np.concatenate([np.full(int(b-a),v) for a,b,v in zip(row['boundaries'][:-1],row['boundaries'][1:],mapped)])
    truth=np.asarray(row['truth']).reshape(-1);records=[]
    names=['FRFTJ','RGPO','VGPO','R-VGPO','ISRJ','DFTJ','NSJ']
    for labels,kind in [(truth,'Reference'),(pred,'Recovered TV')]:
        cuts=np.r_[0,np.flatnonzero(np.diff(labels)!=0)+1,len(labels)]
        for a,b in zip(cuts[:-1],cuts[1:]):
            value=int(labels[a]);records.append(dict(timeline=kind,start=int(a),end=int(b),offline_label=value,mechanism=names[value-1]))
    return records

def timeline_plot(records,path):
    colors=['#0072B2','#E69F00','#009E73','#CC79A7','#56B4E9','#D55E00','#999999']
    names=['FRFTJ','RGPO','VGPO','R-VGPO','ISRJ','DFTJ','NSJ']
    fig,ax=plt.subplots(figsize=(9,2.35))
    for row in records:
        y=1 if row['timeline']=='Reference' else 0
        a,b,value=int(row['start']),int(row['end']),int(row['offline_label'])
        ax.broken_barh([(a,b-a)],(y-.24,.48),facecolors=colors[value-1],edgecolors='white',linewidth=.5)
    ax.set_yticks([0,1],['Recovered TV','Reference']);ax.set_ylim(-.6,1.6)
    ax.set_xlim(0,max(int(r['end']) for r in records));ax.set_xlabel('Retained observation index')
    ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
    fig.legend(handles=[Patch(facecolor=c,label=n) for c,n in zip(colors,names)],loc='upper center',ncol=7,frameon=False,bbox_to_anchor=(.5,1.04),fontsize=8)
    fig.tight_layout(rect=(0,0,1,.90));fig.savefig(path,bbox_inches='tight');plt.close(fig)

def plot_all(output,results):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    source=Path(results)
    sequence_name='paired_summary.csv'
    required=['boundary_summary.csv','ablation_summary.csv','global_ari_bootstrap.csv',sequence_name]
    for name in required:
        if not (source/name).exists():raise FileNotFoundError(f'Missing {source/name}. Run evaluate before plot.')
    boundary=read_csv(source/'boundary_summary.csv');ablation=read_csv(source/'ablation_summary.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42})
    plot_boundary.P=output;plot_boundary.allrows=boundary
    plot_boundary.plot(['LDCR-Net','CPC','PELT','BiLSTM','LSTM-FCN'],['random_loss','receiver_error'],'figure24_unified_boundary.pdf')
    plot_ablation.plot(ablation,output)
    with tempfile.TemporaryDirectory(prefix='plot-data-',dir=output) as directory:
        directory=Path(directory)
        shutil.copy2(source/sequence_name,directory/'paired_summary.csv')
        shutil.copy2(source/'global_ari_bootstrap.csv',directory/'global_ari_bootstrap.csv')
        (directory/'completed_run.json').write_text('{}')
        plot_sequence.HERE=directory;plot_sequence.main()
        shutil.copy2(directory/'paired_four_panel.pdf',output/'figure21_paired_pressure.pdf')
        if (directory/'paired_extended_loss.pdf').exists():
            shutil.copy2(directory/'paired_extended_loss.pdf',output/'figure23_extended_loss.pdf')
    records=timeline_records(source);write_csv(output/'timeline.csv',records)
    timeline_plot(records,output/'figure22_paired_timeline.pdf')
    complexity(output)
    tables=[]
    for table,rows in [('Table 2',boundary),('Table 3',ablation),('Table 4',read_csv(source/sequence_name))]:
        for row in rows:
            if (row['family'],float(row['level'])) not in [('random_loss',0.),('random_loss',.2),('receiver_error',2.)]:continue
            for key in ['precision','recall','f1'] if table!='Table 4' else ['ari_global','normalized_edit_distance','transition_edge_f1','duration_edit_nmae']:
                tables.append(dict(table=table,method=row['method'],family=row['family'],level=row['level'],metric=key,value=row[key]))
    write_csv(output/'paper_tables.csv',tables)
    print('PLOT: empirical figures and representative table values exported to',output,flush=True)

def plot_training(checkpoints,output):
    methods=['full','no_median','no_multiscale','no_reconstruction','no_tv','no_invariance','cpc','bilstm','lstm_fcn']
    fig,axes=plt.subplots(3,3,figsize=(10.5,8))
    for ax,name in zip(axes.flat,methods):
        rows=read_csv(Path(checkpoints)/(name+'_training_history.csv'))
        ax.plot([int(r['epoch']) for r in rows],[float(r['loss']) for r in rows],marker='o',ms=3)
        ax.set(title=name,xlabel='Epoch',ylabel='Training loss');ax.grid(alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(Path(output)/'training_losses.pdf',bbox_inches='tight');plt.close(fig)
