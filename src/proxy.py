import torch
from torch import nn


class Alanine(nn.Module):
    def __init__(self, args, md):
        super().__init__()
        
        self.force = args.force

        self.num_particles = md.num_particles
        self.input_dim = md.num_particles*3
        self.output_dim = md.num_particles*3 if self.force else 1

        self.linear = nn.Linear(self.input_dim, 128)

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

        self.log_z = nn.Parameter(torch.tensor(0.))

        self.to(args.device)

    def forward(self, pos):
        if not self.force:
            pos.requires_grad = True

        pos_ = self.linear(pos.view(-1, self.input_dim))
        out = self.mlp(pos_)

        if not self.force:
            force = - 10 * torch.autograd.grad(out.sum(), pos, create_graph=True, retain_graph=True)[0]
        else:
            force = out.view(*pos.shape)
                
        return force