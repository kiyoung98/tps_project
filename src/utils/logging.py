import os
import sys
import wandb
import torch
import logging

from .plot import *
from .metrics import Metric


class Logger:
    def __init__(self, args, md):
        self.type = args.type
        self.wandb = args.wandb
        self.molecule = args.molecule
        self.start_file = md.start_file
        self.heavy_only = args.heavy_only
        self.save_freq = args.save_freq if args.type == "train" else 1

        self.best_epd = float("inf")
        self.best_elr = float("-inf")
        self.metric = Metric(args, md)

        if args.type == "eval":
            self.save_dir = os.path.join(
                args.save_dir,
                args.project,
                args.date,
                args.type,
                args.best,
                str(args.seed),
            )
        else:
            self.save_dir = os.path.join(
                args.save_dir, args.project, args.date, args.type, str(args.seed)
            )

        for name in [
            "paths",
            "path",
            "potentials",
            "potential",
            "etps",
            "efps",
            "policies",
            "3D_views",
        ]:
            if not os.path.exists(f"{self.save_dir}/{name}"):
                os.makedirs(f"{self.save_dir}/{name}")

        # Logger basic configurations
        self.logger = logging.getLogger("tps")
        self.logger.setLevel(logging.INFO)

        # File handler
        log_file = args.type + ".log"
        log_file = os.path.join(self.save_dir, log_file)
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(levelname)s: %(message)s")
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False

        for k, v in vars(args).items():
            self.logger.info(f"{k}: {v}")

    def info(self, message):
        if self.logger:
            self.logger.info(message)

    def log(
        self,
        policy,
        rollout,
        positions,
        potentials,
        log_md_reward,
        log_target_reward,
        unbiased_md_ll,
        last_idx,
        last_position,
        target_position,
    ):

        # Calculate metrics
        thp, etps, etp_idxs, etp, std_etp, efps, efp_idxs, efp, std_efp = (
            self.metric.cv_metrics(last_idx, last_position, target_position, potentials)
        )
        if self.molecule == "chignolin":
            asp3od_thr6og, asp3n_thr8o = chignolin_h_bond(positions)
            eat36, std_at36 = asp3od_thr6og.mean().item(), asp3od_thr6og.std().item()
            eat38, std_at38 = asp3n_thr8o.mean().item(), asp3n_thr8o.std().item()

        elif self.molecule == "poly":
            h = poly_handed(positions)
            eh, std_h = h.mean().item(), h.std().item()

        ermsd, std_rmsd = self.metric.rmsd(last_position, target_position)
        ll, std_ll = unbiased_md_ll.mean().item(), unbiased_md_ll.std().item()
        epd, std_pd = self.metric.pairwise_distance(last_position, target_position)
        elpd, std_lpd = self.metric.log_pairwise_distance(
            last_position, target_position, self.heavy_only
        )
        epcd, std_pcd = self.metric.pairwise_coulomb_distance(
            last_position, target_position
        )
        len, std_len = last_idx.float().mean().item(), last_idx.float().std().item()

        elmr, std_lmr = log_md_reward.mean().item(), log_md_reward.std().item()
        eltr, std_ltr = log_target_reward.mean().item(), log_target_reward.std().item()
        log_reward = log_md_reward + log_target_reward
        elr, std_lr = log_reward.mean().item(), log_reward.std().item()

        # Log
        if self.type == "train":
            self.logger.info(
                "-----------------------------------------------------------"
            )
            self.logger.info(f"Rollout: {rollout}")
            if epd < self.best_epd:
                self.best_epd = epd
                torch.save(policy.state_dict(), f"{self.save_dir}/epd_policy.pt")
            if elr > self.best_elr:
                self.best_elr = elr
                torch.save(policy.state_dict(), f"{self.save_dir}/elr_policy.pt")

        if self.wandb:
            log = {
                "log_z": policy.log_z.item(),
                "ll": ll,
                "epd": epd,
                "thp": thp,
                "etp": etp,
                "efp": efp,
                "elpd": elpd,
                "epcd": epcd,
                "elr": elr,
                "elmr": elmr,
                "eltr": eltr,
                "ermsd": ermsd,
                "len": len,
                "std_ll": std_ll,
                "std_pd": std_pd,
                "std_lpd": std_lpd,
                "std_pcd": std_pcd,
                "std_etp": std_etp,
                "std_efp": std_efp,
                "std_lr": std_lr,
                "std_lmr": std_lmr,
                "std_ltr": std_ltr,
                "std_rmsd": std_rmsd,
                "std_len": std_len,
            }

            if self.molecule == "chignolin":
                cv_log = {
                    "eat36": eat36,
                    "eat38": eat38,
                    "std_at36": std_at36,
                    "std_at38": std_at38,
                }
                log.update(cv_log)

            elif self.molecule == "poly":
                cv_log = {
                    "eh": eh,
                    "std_h": std_h,
                }
                log.update(cv_log)

            wandb.log(log, step=rollout)

        self.logger.info(f"log_z: {policy.log_z.item()}")
        self.logger.info(f"ll: {ll}")
        self.logger.info(f"epd: {epd}")
        self.logger.info(f"thp: {thp}")
        self.logger.info(f"etp: {etp}")
        self.logger.info(f"efp: {efp}")
        self.logger.info(f"elpd: {elpd}")
        self.logger.info(f"epcd: {epcd}")
        self.logger.info(f"elr: {elr}")
        self.logger.info(f"elmr: {elmr}")
        self.logger.info(f"eltr: {eltr}")
        self.logger.info(f"ermsd: {ermsd}")
        self.logger.info(f"len: {len}")
        self.logger.info(f"std_ll: {std_ll}")
        self.logger.info(f"std_pd: {std_pd}")
        self.logger.info(f"std_etp: {std_etp}")
        self.logger.info(f"std_efp: {std_efp}")
        self.logger.info(f"std_lpd: {std_lpd}")
        self.logger.info(f"std_pcd: {std_pcd}")
        self.logger.info(f"std_lr: {std_lr}")
        self.logger.info(f"std_lmr: {std_lmr}")
        self.logger.info(f"std_ltr: {std_ltr}")
        self.logger.info(f"std_rmsd: {std_rmsd}")
        self.logger.info(f"std_len: {std_len}")

        if self.molecule == "chignolin":
            self.logger.info(f"eat36: {eat36}")
            self.logger.info(f"eat38: {eat38}")
            self.logger.info(f"std_at36: {std_at36}")
            self.logger.info(f"std_at38: {std_at38}")

        elif self.molecule == "poly":
            self.logger.info(f"eh: {eh}")
            self.logger.info(f"std_h: {std_h}")

        if rollout % self.save_freq == 0:
            torch.save(policy.state_dict(), f"{self.save_dir}/policies/{rollout}.pt")

            if self.molecule == "alanine":
                fig_path = plot_paths_alanine(
                    self.save_dir, rollout, positions, target_position, last_idx
                )
            elif self.molecule == "chignolin":
                fig_path = plot_paths_chignolin(
                    self.save_dir, rollout, positions, last_idx
                )
            elif self.molecule == "poly":
                fig_path = plot_hands(self.save_dir, rollout, positions, last_idx)

            fig_pot = plot_potentials(self.save_dir, rollout, potentials, last_idx)

            if self.wandb:
                log = {"potentials": wandb.Image(fig_pot)}

                fig_etp = plot_etps(self.save_dir, rollout, etps, etp_idxs)
                fig_efp = plot_efps(self.save_dir, rollout, efps, efp_idxs)

                cv_log = {
                    "paths": wandb.Image(fig_path),
                    "etps": wandb.Image(fig_etp),
                    "efps": wandb.Image(fig_efp),
                }
                log.update(cv_log)

                wandb.log(log, step=rollout)

        if self.type == "eval":
            if self.molecule == "alanine":
                plot_path_alanine(self.save_dir, positions, target_position, last_idx)
            elif self.molecule == "chignolin":
                plot_path_chignolin(self.save_dir, positions, last_idx)
            elif self.molecule == "poly":
                plot_hand(self.save_dir, positions, last_idx)
            plot_potential(self.save_dir, potentials, last_idx)
            plot_3D_view(
                self.save_dir, self.start_file, positions, potentials, last_idx
            )
