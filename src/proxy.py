import torch
from torch import nn


class Alanine(nn.Module):
    def __init__(self, args, md):
        super().__init__()

        self.force = args.force
        self.feat_aug = args.feat_aug

        self.num_particles = md.num_particles
        if args.feat_aug == "dist":
            self.input_dim = md.num_particles * (3 + 1)
        elif args.feat_aug in ["rel_pos", "norm_rel_pos"]:
            self.input_dim = md.num_particles * (3 + 3)
        else:
            self.input_dim = md.num_particles * 3

        self.output_dim = md.num_particles * 3 if self.force else 1

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim, bias=False),
        )
        self.log_z = nn.Parameter(torch.tensor(args.log_z))

        self.to(args.device)

    def forward(self, pos, target):
        if not self.force:
            pos.requires_grad = True
        if self.feat_aug == "dist":
            dist = torch.norm(pos - target, dim=-1, keepdim=True)
            pos_ = torch.cat([pos, dist], dim=-1)
        elif self.feat_aug == "rel_pos":
            pos_ = torch.cat([pos, pos - target], dim=-1)
        elif self.feat_aug == "norm_rel_pos":
            pos_ = torch.cat(
                [pos, (pos - target) / torch.norm(pos - target, dim=-1, keepdim=True)],
                dim=-1,
            )
        else:
            pos_ = pos

        out = self.mlp(pos_.reshape(-1, self.input_dim))

        if not self.force:
            force = -torch.autograd.grad(
                out.sum(), pos, create_graph=True, retain_graph=True
            )[0]
        else:
            force = out.view(*pos.shape)

        return force


class Chignolin(nn.Module):
    def __init__(self, args, md):
        super().__init__()

        self.force = args.force
        self.feat_aug = args.feat_aug

        self.num_particles = md.num_particles
        if args.feat_aug == "dist":
            self.input_dim = md.num_particles * (3 + 1)
        elif args.feat_aug in ["rel_pos", "norm_rel_pos"]:
            self.input_dim = md.num_particles * (3 + 3)
        else:
            self.input_dim = md.num_particles * 3
        self.output_dim = md.num_particles * 3 if self.force else 1

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim, bias=False),
        )

        self.log_z = nn.Parameter(torch.tensor(args.log_z))

        self.to(args.device)

    def forward(self, pos, target):
        if not self.force:
            pos.requires_grad = True
        if self.feat_aug == "dist":
            dist = torch.norm(pos - target, dim=-1, keepdim=True)
            pos_ = torch.cat([pos, dist], dim=-1)
        elif self.feat_aug == "rel_pos":
            pos_ = torch.cat([pos, pos - target], dim=-1)
        elif self.feat_aug == "norm_rel_pos":
            pos_ = torch.cat(
                [pos, (pos - target) / torch.norm(pos - target, dim=-1, keepdim=True)],
                dim=-1,
            )
        else:
            pos_ = pos

        out = self.mlp(pos_.reshape(-1, self.input_dim))

        if not self.force:
            force = -torch.autograd.grad(
                out.sum(), pos, create_graph=True, retain_graph=True
            )[0]
        else:
            force = out.view(*pos.shape)

        return force


class Poly(nn.Module):
    def __init__(self, args, md):
        super().__init__()

        self.force = args.force

        self.num_particles = md.num_particles
        self.input_dim = md.num_particles * (3 + 1)
        self.output_dim = md.num_particles * 3 if self.force else 1

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim, bias=False),
        )

        self.log_z = nn.Parameter(torch.tensor(args.log_z))

        self.to(args.device)

    def forward(self, pos, target):
        if not self.force:
            pos.requires_grad = True

        dist = torch.norm(pos - target, dim=-1, keepdim=True)
        pos_ = torch.cat([pos, dist], dim=-1)

        out = self.mlp(pos_.reshape(-1, self.input_dim))

        if not self.force:
            force = -torch.autograd.grad(
                out.sum(), pos, create_graph=True, retain_graph=True
            )[0]
        else:
            force = out.view(*pos.shape)

        return force


class Histidine(nn.Module):
    def __init__(self, args, md):
        super().__init__()

        self.force = args.force
        self.feat_aug = args.feat_aug

        self.num_particles = md.num_particles
        if args.feat_aug == "dist":
            self.input_dim = md.num_particles * (3 + 1)
        elif args.feat_aug in ["rel_pos", "norm_rel_pos"]:
            self.input_dim = md.num_particles * (3 + 3)
        else:
            self.input_dim = md.num_particles * 3
        self.output_dim = md.num_particles * 3 if self.force else 1

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim, bias=False),
        )

        self.log_z = nn.Parameter(torch.tensor(0.0))

        self.to(args.device)

    def forward(self, pos, target):
        if not self.force:
            pos.requires_grad = True
        if self.feat_aug == "dist":
            dist = torch.norm(pos - target, dim=-1, keepdim=True)
            pos_ = torch.cat([pos, dist], dim=-1)
        elif self.feat_aug == "rel_pos":
            pos_ = torch.cat([pos, pos - target], dim=-1)
        elif self.feat_aug == "norm_rel_pos":
            pos_ = torch.cat(
                [pos, (pos - target) / torch.norm(pos - target, dim=-1, keepdim=True)],
                dim=-1,
            )
        else:
            pos_ = pos

        out = self.mlp(pos_.reshape(-1, self.input_dim))

        if not self.force:
            force = -torch.autograd.grad(
                out.sum(), pos, create_graph=True, retain_graph=True
            )[0]
        else:
            force = out.view(*pos.shape)

        return force
