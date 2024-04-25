import torch
from torch import nn


class Alanine(nn.Module):
    def __init__(self, args, md):
        super().__init__()
        
        self.force = args.force
        self.goal_conditioned = args.goal_conditioned

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

        if self.goal_conditioned:
            self.goal_linear = nn.Linear(self.input_dim, 128, bias=False)

            self.log_z_mlp = nn.Sequential(
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
        else:
            self.log_z = nn.Parameter(torch.tensor(0.))

        self.to(args.device)

    def forward(self, pos, goal):
        if not self.force:
            pos.requires_grad = True
           
        if self.goal_conditioned:            
            pos_ = self.linear(pos.view(*pos.shape[:-2], -1))
            goal = self.goal_linear(goal.view(*goal.shape[:-2], -1))
            out = self.mlp(pos_+goal)
        else:
            pos_ = self.linear(pos.view(*pos.shape[:-2], -1))
            out = self.mlp(pos_)

        if not self.force:
            force = - self.input_dim / 3 * torch.autograd.grad(out.sum(), pos, create_graph=True, retain_graph=True)[0] # 3 controls bias scale to fit the case where force is true
        else:
            force = out.view(*pos.shape)
                
        return force

    def get_log_z(self, start, goal):
        if self.goal_conditioned:            
            start = self.linear(start.view(*start.shape[:-2], -1))
            goal = self.goal_linear(goal.view(*goal.shape[:-2], -1))
            log_z = self.log_z_mlp(start+goal)
        else:
            log_z = self.log_z
        return log_z
    

class Chignolin(nn.Module):
    def __init__(self, args, md):
        super().__init__()
        
        self.force = args.force
        self.goal_conditioned = args.goal_conditioned

        self.num_particles = md.num_particles
        self.input_dim = md.num_particles*3
        self.output_dim = md.num_particles*3 if self.force else 1

        self.linear = nn.Linear(self.input_dim, 512)

        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim, bias=False)
        )

        if self.goal_conditioned:
            self.goal_linear = nn.Linear(self.input_dim, 512, bias=False)

            self.log_z_mlp = nn.Sequential(
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
            )
        else:
            self.log_z = nn.Parameter(torch.tensor(0.))

        self.to(args.device)

    def forward(self, pos, goal):
        if not self.force:
            pos.requires_grad = True
           
        if self.goal_conditioned:            
            pos_ = self.linear(pos.view(*pos.shape[:-2], -1))
            goal = self.goal_linear(goal.view(*goal.shape[:-2], -1))
            out = self.mlp(pos_+goal)
        else:
            pos_ = self.linear(pos.view(*pos.shape[:-2], -1))
            out = self.mlp(pos_)

        if not self.force:
            force = - self.input_dim / 3 * torch.autograd.grad(out.sum(), pos, create_graph=True, retain_graph=True)[0] # 3 controls bias scale to fit the case where force is true
        else:
            force = out.view(*pos.shape)
                
        return force

    def get_log_z(self, start, goal):
        if self.goal_conditioned:            
            start = self.linear(start.view(*start.shape[:-2], -1))
            goal = self.goal_linear(goal.view(*goal.shape[:-2], -1))
            log_z = self.log_z_mlp(start+goal)
        else:
            log_z = self.log_z
        return log_z
    

class Poly(nn.Module):
    def __init__(self, args, md):
        super().__init__()
        
        self.force = args.force
        self.goal_conditioned = args.goal_conditioned

        self.num_particles = md.num_particles
        self.input_dim = md.num_particles*3
        self.output_dim = md.num_particles*3 if self.force else 1

        self.linear = nn.Linear(self.input_dim, 256)

        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim, bias=False)
        )

        if self.goal_conditioned:
            self.goal_linear = nn.Linear(self.input_dim, 256, bias=False)

            self.log_z_mlp = nn.Sequential(
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
        else:
            self.log_z = nn.Parameter(torch.tensor(0.))

        self.to(args.device)

    def forward(self, pos, goal):
        if not self.force:
            pos.requires_grad = True
           
        if self.goal_conditioned:            
            pos_ = self.linear(pos.view(*pos.shape[:-2], -1))
            goal = self.goal_linear(goal.view(*goal.shape[:-2], -1))
            out = self.mlp(pos_+goal)
        else:
            pos_ = self.linear(pos.view(*pos.shape[:-2], -1))
            out = self.mlp(pos_)

        if not self.force:
            force = - self.input_dim / 3 * torch.autograd.grad(out.sum(), pos, create_graph=True, retain_graph=True)[0] # 3 controls bias scale to fit the case where force is true
        else:
            force = out.view(*pos.shape)
                
        return force

    def get_log_z(self, start, goal):
        if self.goal_conditioned:            
            start = self.linear(start.view(*start.shape[:-2], -1))
            goal = self.goal_linear(goal.view(*goal.shape[:-2], -1))
            log_z = self.log_z_mlp(start+goal)
        else:
            log_z = self.log_z
        return log_z