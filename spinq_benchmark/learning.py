"""CPU Noise2Noise learner for independent complex FID acquisitions."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np


def train_complex_denoiser(train_pairs, validation_pairs, model_path: Path, *, seed=42, epochs=30):
    import torch
    from torch import nn
    torch.set_num_threads(min(torch.get_num_threads(),4))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if len(train_pairs)<4 or len(validation_pairs)<2:
        raise ValueError("Not enough independent acquisition pairs")
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
    return {"scale":scale,"validation_loss":best,"epochs_executed":epoch+1,
            "train_pairs":len(train_pairs),"validation_pairs":len(validation_pairs)}


def apply_complex_denoiser(signal, model_path: Path):
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
