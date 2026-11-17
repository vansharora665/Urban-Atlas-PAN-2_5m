"""Pointwise texture student: no positional embeddings, dropout disabled at test."""

import torch


class PanTextureNet(torch.nn.Module):
    def __init__(self,mean,scale,hidden=(192,128)):
        super().__init__()
        self.register_buffer("mean",torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer("scale",torch.as_tensor(scale,dtype=torch.float32).clamp_min(1e-4))
        self.layers=torch.nn.Sequential(torch.nn.Linear(16,hidden[0]),torch.nn.GELU(),torch.nn.Dropout(.03),
                                        torch.nn.Linear(hidden[0],hidden[1]),torch.nn.GELU(),torch.nn.Linear(hidden[1],4))

    def forward(self,features):
        return self.layers(((features-self.mean)/self.scale).clamp(-8,8))
