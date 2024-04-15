import torch
from torch import nn


class Alanine(nn.Module):
    def __init__(self, args, md_info):
        super().__init__()
        
        self.force = args.force
        self.goal_conditioned = args.goal_conditioned

        self.num_particles = md_info.num_particles
        self.input_dim = md_info.num_particles*3
        self.output_dim = md_info.num_particles*3 if self.force else 1

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
            nn.Linear(128, self.output_dim, bias=args.bias)
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
            pos_, rot_inv = self.canonicalize(pos)
            goal = self.canonicalize(goal)[0]
            
            pos_ = self.linear(pos_.view(*pos.shape[:-2], -1))
            goal = self.goal_linear(goal.view(*goal.shape[:-2], -1))
            out = self.mlp(pos_+goal)
        else:
            pos_ = self.linear(pos.view(*pos.shape[:-2], -1))
            out = self.mlp(pos_)

        if not self.force:
            force = - self.input_dim / 6 * torch.autograd.grad(out.sum(), pos, create_graph=True, retain_graph=True)[0] # 6 controls bias scale to fit the case where force is true
        else:
            if self.goal_conditioned:
                out = out.view(-1, self.num_particles, 3)
                rot_inv = rot_inv.view(-1, 3, 3)
                force = torch.bmm(out, rot_inv).view(*pos.shape)
            else:
                force = out.view(*pos.shape)
                
        return force

    def get_log_z(self, start, goal):
        if self.goal_conditioned:
            start = self.canonicalize(start)[0]
            goal = self.canonicalize(goal)[0]
            
            start = self.linear(start.view(*start.shape[:-2], -1))
            goal = self.goal_linear(goal.view(*goal.shape[:-2], -1))
            log_z = self.log_z_mlp(start+goal)
        else:
            log_z = self.log_z
        return log_z
    
    def canonicalize(self, pos):        
        N = pos[:, :, 6].detach()
        CA = pos[:, :, 8].detach()
        C = pos[:, :, 14].detach()
        
        pos_ = pos - CA.unsqueeze(-2)

        x = N - CA
        y = C - CA
        z = torch.cross(x, y)
        y = y / torch.norm(y, dim=-1, keepdim=True)
        z = z / torch.norm(z, dim=-1, keepdim=True)
        x = torch.cross(y, z)

        rot = torch.stack([x, y, z], -1)
        rot_inv = torch.stack([x, y, z], -2)

        pos_ = pos_.view(-1, self.num_particles, 3)
        rot = rot.view(-1, 3, 3)

        pos_ = torch.bmm(pos_, rot).view(*pos.shape)
        return pos_, rot_inv
    

# TODO: goal_condition is not implemented
class Chignolin(nn.Module):    
    def __init__(self, args, md_info):
        super().__init__()
        
        self.force = args.force

        self.input_dim = md_info.num_particles*3
        self.output_dim = md_info.num_particles*3 if self.force else 1

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
            nn.Linear(512, self.output_dim, bias=args.bias)
        )

        self.log_z = nn.Parameter(torch.tensor(0.))

        self.to(args.device)

    def forward(self, pos, goal):
        if not self.force:
            pos.requires_grad = True
           
        pos_ = self.linear(pos)
        out = self.mlp(pos_)

        if not self.force:
            force = - self.input_dim / 6 * torch.autograd.grad(out.sum(), pos, create_graph=True, retain_graph=True)[0] 
        else:
            force = out
                
        return force

    def get_log_z(self, start, goal):
        return self.log_z
    

# TODO: goal_condition is not implemented
class Poly(nn.Module):
    def __init__(self, args, md_info):
        super().__init__()
        
        self.force = args.force

        self.input_dim = md_info.num_particles*3
        self.output_dim = md_info.num_particles*3 if self.force else 1

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
            nn.Linear(256, self.output_dim, bias=args.bias)
        )

        self.log_z = nn.Parameter(torch.tensor(0.))

        self.to(args.device)

    def forward(self, pos, goal):
        if not self.force:
            pos.requires_grad = True
           
        pos_ = self.linear(pos)
        out = self.mlp(pos_)

        if not self.force:
            force = - self.input_dim / 3 * torch.autograd.grad(out.sum(), pos, create_graph=True, retain_graph=True)[0] 
        else:
            force = out
                
        return force

    def get_log_z(self, start, goal):
        return self.log_z