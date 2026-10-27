from pathlib import Path
import matplotlib.pyplot as plt
def plot(records,output):
    root=Path(output);qa=root
    methods = ['LDCR-Net', 'No median', 'No multiscale', 'No reconstruction',
               'No TV', 'No invariance']
    labels = ['LDCR-Net', 'w/o median', 'w/o multiscale', 'w/o reconstruction',
              'w/o TV', 'w/o cross-view consistency']
    colors = ['#1261A0', '#E58025', '#208C70', '#8855A8', '#A73244', '#656565']
    markers = ['o', 's', '^', 'D', 'v', 'P']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'pdf.fonttype': 42})
    plotted = []
    for family, output in [('random_loss', 'figure25_unified_ablation_loss'),
                           ('receiver_error', 'figure26_unified_ablation_error')]:
        fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.25))
        for col, metric in enumerate(['precision', 'recall', 'f1']):
            ax = axes[col]
            for j, method in enumerate(methods):
                group = sorted([r for r in records if r['family'] == family and
                                r['method'] == method], key=lambda r: float(r['level']))
                assert group
                x = [float(r['level']) * (100 if family == 'random_loss' else 1)
                     for r in group]
                y = [float(r[metric]) for r in group]
                plotted.extend({'family': family, 'method': method, 'metric': metric,
                                'level': float(r['level']), 'value': float(r[metric])}
                               for r in group)
                ax.plot(x, y, color=colors[j], marker=markers[j], ms=4, lw=1.5,
                        ls='--' if method == 'No invariance' else '-', label=labels[j])
            ax.set(xlabel='Observation loss (%)' if family == 'random_loss'
                   else 'Parameter-error scale', ylabel={'f1': 'F1', 'precision':
                   'Precision', 'recall': 'Recall'}[metric], ylim=(0, 1.04))
            ax.set_xticks(x)
            ax.grid(alpha=.18)
            ax.spines[['top', 'right']].set_visible(False)
            ax.text(.02, .06, f'({chr(97 + col)})', transform=ax.transAxes)
        handles, names = axes[0].get_legend_handles_labels()
        fig.legend(handles, names, loc='upper center', ncol=3, frameon=False,
                   bbox_to_anchor=(.5, 1.015))
        fig.tight_layout(rect=(0, 0, 1, .83))
        fig.savefig(root / (output + '.pdf'), bbox_inches='tight')
        fig.savefig(qa / (output + '.png'), bbox_inches='tight', dpi=120)
        plt.close(fig)
    
    assert len(plotted) == 3 * len(records)
