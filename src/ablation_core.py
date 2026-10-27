import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from detector_contract import check_detector


CONDITIONS=[('random_loss',v) for v in [0.,.05,.1,.15,.2]]+[
            ('receiver_error',v) for v in [0.,.5,1.,1.5,2.]]


def predict(model,s,settings):
    check_detector(settings,model.output_stride)
    with torch.inference_mode():
        out=model(torch.as_tensor(np.asarray(s['pdw_data'],np.float32).T.copy())[None])
        z=out['z_ti'][0]
        v=torch.stack([model.dual_window_score(z,w) for w in settings['windows']]).clamp_min(1e-8).log().mean(0).exp().numpy()
    peaks,_=find_peaks(gaussian_filter1d(v,settings['sigma'],mode='nearest'),
       prominence=settings['prominence'],distance=settings['distance'],width=settings['width'])
    return sorted(set(model.latent_to_input_index(int(k)) for k in peaks
                      if 0<model.latent_to_input_index(int(k))<len(s['pdw_data'])))
