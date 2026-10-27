"""Exact primary paired inference, frozen grouping and sequence aggregation."""
from pathlib import Path
import csv
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import adjusted_rand_score
from sequence_metrics import expand_timeline,score_timelines
HERE=Path('.')


MODES=['tv','ti','pdw']


METRICS=['ari_sequence','transition_edge_f1','normalized_edit_distance','duration_edit_nmae','pulse_error_rate']


LOWER=np.array([0.,-10.,0.,-15.])


UPPER=np.array([180.,10.,4.,30.])


def save_csv(name, rows):
    if not rows:return
    with (HERE/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def pool(model, latent, boundaries):
    values=[]
    for start,end in zip(boundaries[:-1],boundaries[1:]):
        left=model.input_to_latent_index(int(start),len(latent))
        right=model.input_to_latent_index(int(end),len(latent))
        if right-left>2:left+=1;right-=1
        right=max(left+1,right);part=latent[left:right]
        if not len(part):part=latent[-1:]
        values.append(torch.cat([part.mean(0),part.std(0,unbiased=False)]).cpu().numpy())
    return np.asarray(values)


def extract(model,sample,device,oracle=False):
    pdw=np.asarray(sample['pdw_data'],np.float32)
    with torch.inference_mode():
        out=model(torch.as_tensor(pdw.T.copy(),device=device)[None])
        z=out['z_ti'][0]
        score=torch.stack([model.dual_window_score(z,w) for w in (6,8,10)]).clamp_min(1e-8).log().mean(0).exp().cpu().numpy()
        peaks,_=find_peaks(gaussian_filter1d(score,1.,mode='nearest'),prominence=.30,distance=25,width=1.)
        detected=sorted(set(int(v)*2 for v in peaks if 0<int(v)*2<len(pdw)))
        cuts=list(map(int,sample['change_points'])) if oracle else detected
        boundaries=np.array([0]+cuts+[len(pdw)],dtype=int)
        features={'ti':pool(model,out['z_ti'][0],boundaries),
                  'tv':pool(model,torch.nn.functional.normalize(out['z_tv'][0],dim=-1),boundaries)}
    scaled=np.clip(2*(pdw-LOWER)/(UPPER-LOWER)-1,-3,3)
    features['pdw']=np.asarray([np.r_[scaled[a:b].mean(0),scaled[a:b].std(0)] for a,b in zip(boundaries[:-1],boundaries[1:])])
    labels=np.asarray(sample['labels']).reshape(-1)
    semantics=np.asarray([np.bincount(labels[a:b]).argmax() for a,b in zip(boundaries[:-1],boundaries[1:])])
    return {'features':features,'boundaries':boundaries,'semantics':semantics,'truth':labels,
            'detected_cuts':detected,'true_cuts':np.asarray(sample['change_points']),
            'id':int(sample['sample_id']),'seed':int(sample['generation_seed'])}


def fit_references(rows):
    references={}
    truth=np.concatenate([r['semantics'] for r in rows])
    for mode in MODES:
        x=normalize(np.concatenate([r['features'][mode] for r in rows]))
        kmeans=KMeans(7,n_init=20,random_state=42).fit(x)
        overlap=np.array([[sum((kmeans.labels_==k)&(truth==label)) for label in range(1,8)] for k in range(7)])
        a,b=linear_sum_assignment(-overlap)
        references[mode]=(kmeans,{int(k):int(label)+1 for k,label in zip(a,b)})
        np.savez_compressed(HERE/(mode+'_frozen_reference.npz'),centers=kmeans.cluster_centers_,
                            scoring_mapping=np.array(references[mode][1],dtype=object),overlap=overlap)
    return references


def score_rows(rows,references,family,level):
    results=[];alltruth=np.concatenate([r['semantics'] for r in rows]);grouping={}
    for mode,(reference,mapping) in references.items():
        states=[]
        for row in rows:
            assigned=reference.predict(normalize(row['features'][mode]).astype(reference.cluster_centers_.dtype))
            states.append(assigned)
            mapped=np.array([mapping[int(s)] for s in assigned])
            predicted=expand_timeline(mapped,row['boundaries'])
            metrics=score_timelines(row['truth'],predicted)
            metrics['ari_sequence']=float(adjusted_rand_score(row['semantics'],assigned))
            results.append({'family':family,'level':level,'method':mode,'sample':row['id'],'seed':row['seed'],**metrics})
        grouping[mode]=float(adjusted_rand_score(alltruth,np.concatenate(states)))
    return results,grouping


def summarize(results,grouping):
    summary=[];differences=[];rng=np.random.default_rng(86017)
    groups=list(dict.fromkeys((r['family'],r['level']) for r in results))
    for family,level in groups:
        bymode={mode:[r for r in results if (r['family'],r['level'],r['method'])==(family,level,mode)] for mode in MODES}
        n=len(bymode['tv']);boot=rng.integers(0,n,(2000,n))
        for mode,rows in bymode.items():
            sr={'family':family,'level':level,'method':mode,'num_sequences':n,
                'ari_global':grouping[family,level][mode]}
            for metric in METRICS:
                values=np.array([r[metric] for r in rows]);lo,hi=np.quantile(values[boot].mean(1),[.025,.975])
                sr[metric]=float(values.mean());sr[metric+'_low']=float(lo);sr[metric+'_high']=float(hi)
            summary.append(sr)
        for other in ['ti','pdw']:
            for metric in ['ari_sequence','transition_edge_f1']:
                values=np.array([a[metric]-b[metric] for a,b in zip(bymode['tv'],bymode[other])])
                lo,hi=np.quantile(values[boot].mean(1),[.025,.975])
                differences.append({'family':family,'level':level,'contrast':'tv - '+other,'metric':metric,
                                    'mean_difference':float(values.mean()),'ci_low':float(lo),'ci_high':float(hi)})
    save_csv('paired_summary.csv',summary);save_csv('paired_method_differences.csv',differences)
    save_csv('paired_per_sequence.csv',results)
    # The original graphic uses global ARI. Bootstrap bands are only attached to
    # mean sequence-edge F1; the separately exported per-sequence ARI is explicit.
    transitions=[]
    for family in ['random_loss','receiver_error']:
        levels=sorted(set(r['level'] for r in summary if r['family']==family))
        for mode in MODES:
            rows=[next(r for r in summary if (r['family'],r['level'],r['method'])==(family,l,mode)) for l in levels]
            for lower,higher in zip(rows,rows[1:]):
                before=[r for r in results if (r['family'],r['level'],r['method'])==(family,lower['level'],mode)]
                after=[r for r in results if (r['family'],r['level'],r['method'])==(family,higher['level'],mode)]
                for metric in ['ari_sequence','transition_edge_f1']:
                    values=np.array([b[metric]-a[metric] for a,b in zip(before,after)])
                    boot=rng.integers(0,len(values),(2000,len(values)))
                    lo,hi=np.quantile(values[boot].mean(1),[.025,.975])
                    transitions.append({'family':family,'method':mode,'from_level':lower['level'],
                                        'to_level':higher['level'],'metric':metric,
                                        'mean_difference':float(values.mean()),'ci_low':float(lo),'ci_high':float(hi)})
    save_csv('paired_level_differences.csv',transitions)
    return summary
