from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
P=Path('.')
allrows=[]
colors=['#1261A0','#E58025','#208C70','#8855A8','#A73244','#656565']
markers=['o','s','^','D','v','P']

def plot(groups,families,name):
    fig,axes=plt.subplots(len(families),3,figsize=(10.6,2.9*len(families)),squeeze=False)
    for row,family in enumerate(families):
        for col,key in enumerate(['precision','recall','f1']):
            ax=axes[row,col]
            for j,method in enumerate(groups):
                rr=sorted([r for r in allrows if r['method']==method and r['family']==family],key=lambda r:float(r['level']))
                xx=np.array([float(r['level']) for r in rr]);xx=xx*100 if family=='random_loss' else xx
                yy=np.array([float(r[key]) for r in rr])
                label={'No median':'w/o median','No multiscale':'w/o multiscale','No reconstruction':'w/o reconstruction','No TV':'w/o TV','No consistency':'w/o consistency'}.get(method,method)
                ax.plot(xx,yy,color=colors[j],marker=markers[j],ms=4,lw=1.5,ls='--' if method=='No consistency' else '-',label=label)
            ax.set(xlabel='Observation loss (%)' if family=='random_loss' else 'Parameter-error scale',ylabel={'f1':'F1','precision':'Precision','recall':'Recall'}[key],ylim=(0,1.04))
            ax.set_xticks(xx);ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
            ax.text(.02,.06,f'({chr(97+row*3+col)})',transform=ax.transAxes)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=len(groups),frameon=False,bbox_to_anchor=(.5,1.005))
    fig.tight_layout(rect=(0,0,1,.93 if len(families)==1 else .96));fig.savefig(P/name,bbox_inches='tight');plt.close(fig)
