import torch
from torch import nn


class Alanine(nn.Module):
    def __init__(self, args, md):
        super().__init__()
        
        self.force = args.force

        self.num_particles = md.num_particles
        self.input_dim = md.num_particles*3
        self.output_dim = md.num_particles*3 if self.force else 1

        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim, bias=False)
        )

        self.pos_linear = nn.Linear(self.input_dim, 128)
        self.time_linear = nn.Linear(1, 128)

        self.log_z = nn.Parameter(torch.tensor(0.))

        self.to(args.device)

    def forward(self, pos, t):
        if not self.force:
            pos.requires_grad = True
        
        pos_ = self.pos_linear(pos.reshape(*pos.shape[:-2], self.input_dim))
        t_ = self.time_linear(t)
            
        out = self.mlp(pos_+t_)

        if not self.force:
            force = - torch.autograd.grad(out.sum(), pos, create_graph=True, retain_graph=True)[0]
        else:
            force = out.view(*pos.shape)
                
        return force