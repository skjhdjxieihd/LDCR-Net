import numpy as np
import torch


def alignment(model):
    scale,offset=1,0.
    for layer in model.enc:
        stride=getattr(layer,'stride',1)
        if isinstance(stride,tuple):stride=stride[0]
        kernel=getattr(layer,'kernel_size',1)
        if isinstance(kernel,tuple):kernel=kernel[0]
        offset+=(kernel-1)/2*scale
        scale*=stride
    return scale,int(round(offset))


def cpc_scores(model,pdw):
    values=torch.from_numpy(np.asarray(pdw,dtype=np.float32)).T[None]
    with torch.inference_mode():
        latent=model(values,feature=True)
        scores=model.score(latent[:,:-1],latent[:,1:]).squeeze(0).numpy()
    return scores
