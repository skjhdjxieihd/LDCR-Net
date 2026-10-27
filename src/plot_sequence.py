"""Render all locked outcomes, including nonmonotonic or unfavorable points."""
from pathlib import Path
import os,csv,json,hashlib
HERE=Path(__file__).resolve().parent

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

def read(name):
    with (HERE/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(name,rows):
    with (HERE/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    assert (HERE/'completed_run.json').exists(),'Full declared run must finish first'
    rows=read('paired_summary.csv')
    global_ci={(r['family'],float(r['level']),r['method']):r for r in read('global_ari_bootstrap.csv')}
    expected={(r['family'],float(r['level']),r['method']) for r in rows}
    assert set(global_ci)==expected,'Bootstrap records do not match the evaluated conditions'
    colors={'tv':'#0072B2','ti':'#D55E00','pdw':'#009E73'}
    names={'tv':'TV (proposed)','ti':'TI pooling','pdw':'RPDW pooling'}
    plt.rcParams.update({'font.size':10,'axes.spines.right':False,'axes.spines.top':False,
                         'svg.fonttype':'none','figure.dpi':140})
    def panel(ax,family,metric,max_level):
        for mode in ['tv','ti','pdw']:
            selected=sorted([r for r in rows if r['family']==family and r['method']==mode and float(r['level'])<=max_level],key=lambda r:float(r['level']))
            x=np.array([float(r['level']) for r in selected])*(100 if family=='random_loss' else 1)
            y=[float(r[metric]) for r in selected]
            ax.plot(x,y,'o-',color=colors[mode],label=names[mode],lw=1.8,ms=4)
            if metric=='transition_edge_f1':
                ax.fill_between(x,[float(r[metric+'_low']) for r in selected],
                                  [float(r[metric+'_high']) for r in selected],color=colors[mode],alpha=.12)
            else:
                ci=[global_ci[family,float(r['level']),mode] for r in selected]
                ax.fill_between(x,[float(r['ci_low']) for r in ci],
                                  [float(r['ci_high']) for r in ci],color=colors[mode],alpha=.12)
        ax.set_ylim(0,1.025);ax.grid(alpha=.18)
        ax.set_ylabel('Global clustering ARI' if metric=='ari_global' else 'Mean transition-edge F1')
        ax.set_xlabel('Random observation loss (%)' if family=='random_loss' else 'Receiver-error scale')
        ax.set_xticks([0,5,10,15,20] if family=='random_loss' and max_level==.2 else
                      [0,10,20,30,40,50,60] if family=='random_loss' else [0,.5,1,1.5,2])
    fig,axes=plt.subplots(2,2,figsize=(9,6.6))
    for i,metric in enumerate(['ari_global','transition_edge_f1']):
        panel(axes[i,0],'random_loss',metric,.2);panel(axes[i,1],'receiver_error',metric,2.)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,bbox_to_anchor=(.5,.98),frameon=False)
    fig.suptitle('Paired pressure test with frozen nominal prototypes',y=1.015,fontsize=13)
    fig.tight_layout(rect=(0,.02,1,.94))
    count=int(rows[0]['num_sequences'])
    fig.text(.5,.003,f'{count} paired test sequences; frozen encoder/prototypes; bands: 95% whole-sequence bootstrap.',ha='center',fontsize=8)
    for suffix in ['pdf']:fig.savefig(HERE/('paired_four_panel.'+suffix),bbox_inches='tight',dpi=220)
    plt.close(fig)
    if max(float(r['level']) for r in rows if r['family']=='random_loss')<=.2:return
    fig,axes=plt.subplots(1,2,figsize=(9,3.6))
    for ax,metric in zip(axes,['ari_global','transition_edge_f1']):panel(ax,'random_loss',metric,.6)
    fig.legend(handles,labels,loc='upper center',ncol=3,bbox_to_anchor=(.5,1.04),frameon=False)
    fig.tight_layout(rect=(0,0,1,.92))
    for suffix in ['pdf']:fig.savefig(HERE/('paired_extended_loss.'+suffix),bbox_inches='tight',dpi=220)
    plt.close(fig)
