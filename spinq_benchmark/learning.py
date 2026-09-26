"""CPU Noise2Noise learner for independent complex FID acquisitions."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _fir_windows(signal, taps=9):
    y=np.asarray(signal,dtype=np.complex128)
    if len(y)<taps: raise ValueError("FID too short for learned FIR filter")
    return sliding_window_view(np.pad(y,(taps//2,taps//2),mode="reflect"),taps)


def _train_numpy_fir(train_pairs,validation_pairs,model_path,seed,torch_error):
    """Complex linear Noise2Noise fallback; phase rotation equivariant by design."""
    rng=np.random.default_rng(seed)
    scale=float(np.median([np.sqrt(np.mean(np.abs(x)**2)) for x,_ in train_pairs]))
    if not np.isfinite(scale) or scale<=0: raise ValueError("Invalid training signal scale")
    gram=np.zeros((9,9),dtype=np.complex128)
    target=np.zeros(9,dtype=np.complex128)
    count=0
    for source,other in train_pairs:
        length=min(len(source),len(other))
        if length<256: continue
        x=_fir_windows(np.asarray(source)[:length]/scale)
        y=np.asarray(other)[:length]/scale
        ix=rng.choice(length,size=min(length,4096),replace=False)
        sample=x[ix]
        gram+=sample.conj().T@sample
        target+=sample.conj().T@y[ix]
        count+=len(ix)
    if count==0: raise ValueError("No usable independent training pairs")
    best=None
    for ridge in (0.,1e-5,1e-4,1e-3,1e-2,1e-1):
        taps=np.linalg.solve(gram+max(ridge,1e-10)*count*np.eye(9),target)
        losses=[]
        for source,other in validation_pairs:
            length=min(len(source),len(other))
            if length<256: continue
            x=_fir_windows(np.asarray(source)[:length]/scale)
            y=np.asarray(other)[:length]/scale
            ix=np.linspace(0,length-1,min(length,4096),dtype=int)
            losses.append(float(np.mean(np.abs(x[ix]@taps-y[ix])**2)))
        if not losses: raise ValueError("No usable independent validation pairs")
        loss=float(np.mean(losses))
        if best is None or loss<best[0]: best=(loss,ridge,taps)
    model=model_path.with_suffix(".npz")
    model.parent.mkdir(parents=True,exist_ok=True)
    np.savez(model,taps=best[2],scale=scale,ridge=best[1],validation_loss=best[0])
    return {"backend":"numpy_complex_fir_noise2noise","model_path":str(model),
            "scale":scale,"validation_loss":best[0],"ridge":best[1],
            "train_pairs":len(train_pairs),"validation_pairs":len(validation_pairs),
            "fallback_reason":f"PyTorch unavailable: {type(torch_error).__name__}: {torch_error}"}


def train_complex_denoiser(train_pairs, validation_pairs, model_path: Path, *, seed=42, epochs=30):
    if len(train_pairs)<4 or len(validation_pairs)<2:
        raise ValueError("Not enough independent acquisition pairs")
    try:
        import torch
        from torch import nn
    except (ImportError,OSError,RuntimeError) as exc:
        return _train_numpy_fir(train_pairs,validation_pairs,model_path,seed,exc)
    torch.set_num_threads(min(torch.get_num_threads(),4))
    torch.manual_seed(seed)
    np.random.seed(seed)
    scale=float(np.median([np.sqrt(np.mean(np.abs(x)**2)) for x,_ in train_pairs]))
    if not np.isfinite(scale) or scale<=0: raise ValueError("Invalid training signal scale")
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers=nn.Sequential(nn.Conv1d(2,24,9,padding=4),nn.GELU(),
                nn.Conv1d(24,24,9,padding=4),nn.GELU(),nn.Conv1d(24,2,9,padding=4))
        def forward(self,x):return x+self.layers(x)
    net=Net().cpu()
    opt=torch.optim.Adam(net.parameters(),lr=0.001)
    def tensor(z):
        return torch.as_tensor(np.stack((z.real,z.imag)).copy()/scale,dtype=torch.float32)[None]
    best=float("inf"); best_state=None; patience=0
    for epoch in range(epochs):
        net.train()
        for source,target in train_pairs:
            length=min(len(source),len(target))
            if length<256: continue
            start=np.random.randint(0,max(1,length-1024+1))
            end=min(length,start+1024)
            angle=np.random.uniform(-math.pi,math.pi)
            rot=np.exp(1j*angle)
            x=tensor(source[start:end]*rot); y=tensor(target[start:end]*rot)
            opt.zero_grad()
            loss=((net(x)-y)**2).mean()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val=float(np.mean([((net(tensor(x[:1024]))-tensor(y[:1024]))**2).mean().item()
                               for x,y in validation_pairs]))
        if val<best:
            best=val; best_state=copy.deepcopy(net.state_dict()); patience=0
        else:
            patience+=1
            if patience>=8:break
    if best_state is None: raise ValueError("Neural training did not converge")
    net.load_state_dict(best_state)
    model_path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":best_state,"scale":scale,"validation_loss":best,
                "train_pairs":len(train_pairs),"validation_pairs":len(validation_pairs),
                "epochs_executed":epoch+1,"architecture":"2-24-24-2 Conv1d residual"},model_path)
    return {"backend":"torch_noise2noise","model_path":str(model_path),
            "scale":scale,"validation_loss":best,"epochs_executed":epoch+1,
            "train_pairs":len(train_pairs),"validation_pairs":len(validation_pairs)}


def apply_complex_denoiser(signal, model_path: Path):
    if model_path.suffix==".npz":
        with np.load(model_path,allow_pickle=False) as model:
            taps=model["taps"]
        return _fir_windows(signal)@taps
    import torch
    from torch import nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers=nn.Sequential(nn.Conv1d(2,24,9,padding=4),nn.GELU(),
                nn.Conv1d(24,24,9,padding=4),nn.GELU(),nn.Conv1d(24,2,9,padding=4))
        def forward(self,x):return x+self.layers(x)
    saved=torch.load(model_path,map_location="cpu",weights_only=True)
    net=Net(); net.load_state_dict(saved["state_dict"]);net.eval()
    y=np.asarray(signal,complex)
    scale=saved["scale"]
    x=torch.as_tensor(np.stack((y.real,y.imag)).copy()/scale,dtype=torch.float32)[None]
    with torch.no_grad(): a=net(x).numpy()[0]
    return scale*(a[0]+1j*a[1])
