from easydict import EasyDict as edict
from pprint import PrettyPrinter as PP

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from base_config import flatten_json_iterative_solution
from data import CLEVRMidrepsDataset
from backbones import RNEncoder, RNDecoder, RNDiscriminator
from base_pl_model import BasePLModel
from utils import task_to_out_nc, get_acc_at
from losses import bce_fill

pp = PP(indent=4)


class PLModel(BasePLModel):
    def __init__(self, cfg=None):
        if not isinstance(cfg.tasks, list):
            cfg.tasks = [cfg.tasks]

        super().__init__(cfg)

        self.many_validation = True
        self.encoder = RNEncoder(**cfg.model.encoder.kwargs)
        self.decoder = RNDecoder(tasks=cfg.tasks, **cfg.model.decoder.kwargs)

        self.discriminators = None
        if cfg.model.discriminator.use:
            self.discriminators = nn.ModuleDict()
            for task in cfg.tasks:
                self.discriminators[task] = RNDiscriminator(
                    in_nc=3 + task_to_out_nc[task], **cfg.model.discriminator.kwargs,
                )

        if cfg.train.lnorm == "l1":
            self.lnorm = nn.L1Loss(reduction="none")
        elif cfg.train.lnorm == "l2":
            self.lnorm = nn.MSELoss(reduction="none")
        elif cfg.train.lnorm == "smooth_l1":
            self.lnorm = nn.SmoothL1Loss(reduction="none")
        else:
            print(f"Unkown lnorm: {cfg.train.lnorm}")

        intra_task_normalizer = cfg.train.lnorm_mult
        if cfg.model.discriminator.use:
            intra_task_normalizer += cfg.train.adv_mult
        self.lambdas = edict()
        self.lambdas.lnorm = cfg.train.lnorm_mult / intra_task_normalizer
        self.lambdas.adv = cfg.train.adv_mult / intra_task_normalizer

        inter_task_normalizar = sum(
            [cfg.train.task_lambdas[task] for task in cfg.tasks]
        )
        for task in cfg.tasks:
            self.lambdas[task] = cfg.train.task_lambdas[task] / inter_task_normalizar

        print("Multitask weights:")
        print(self.lambdas)

    def forward(self, img):
        features = self.encoder(img)
        midreps = self.decoder(features)
        return features, midreps

    def configure_optimizers(self):
        encoder_decoder_params = list(self.encoder.parameters())
        encoder_decoder_params += list(self.decoder.parameters())

        def make_opt(parameters, name):
            return torch.optim.Adam(
                parameters,
                lr=getattr(self.cfg.train.lrs, name, self.cfg.train.lrs.default),
                weight_decay=getattr(
                    self.cfg.train.weight_decay,
                    name,
                    self.cfg.train.weight_decay.default,
                ),
            )

        encoder_decoder_opt = make_opt(encoder_decoder_params, "encoder_decoder")
        encoder_decoder_sch = torch.optim.lr_scheduler.MultiStepLR(
            encoder_decoder_opt, milestones=[10, 20], gamma=0.1,
        )
        if self.discriminators:
            discriminator_opt = make_opt(
                self.discriminators.parameters(), "discriminator"
            )
            # return [encoder_decoder_opt, discriminator_opt], [encoder_decoder_sch]
            return [encoder_decoder_opt, discriminator_opt]

        # return [encoder_decoder_opt], [encoder_decoder_sch]
        return [encoder_decoder_opt]

    def prepare_data(self):
        self.orig_train_dataset = CLEVRMidrepsDataset(
            base_dir=self.cfg.orig_dir,
            split="train",
            midreps=self.cfg.tasks,
            transform=CLEVRMidrepsDataset.std_img_transform,
            midreps_transform=CLEVRMidrepsDataset.std_midreps_transforms,
        )

        self.orig_val_dataset = CLEVRMidrepsDataset(
            base_dir=self.cfg.orig_dir,
            split="val",
            midreps=self.cfg.tasks,
            transform=CLEVRMidrepsDataset.std_img_transform,
            midreps_transform=CLEVRMidrepsDataset.std_midreps_transforms,
        )

        self.uni_train_dataset = CLEVRMidrepsDataset(
            base_dir=self.cfg.uni_dir,
            split="train",
            midreps=self.cfg.tasks,
            transform=CLEVRMidrepsDataset.std_img_transform,
            midreps_transform=CLEVRMidrepsDataset.std_midreps_transforms,
        )

        self.uni_val_dataset = CLEVRMidrepsDataset(
            base_dir=self.cfg.uni_dir,
            split="val",
            midreps=self.cfg.tasks,
            transform=CLEVRMidrepsDataset.std_img_transform,
            midreps_transform=CLEVRMidrepsDataset.std_midreps_transforms,
        )

    def train_dataloader(self):
        return DataLoader(
            self.orig_train_dataset,
            shuffle=True,
            drop_last=True,
            batch_size=self.cfg.train.bsz,
            num_workers=self.cfg.num_workers,
            pin_memory=self.use_cuda,
        )

    def val_dataloader(self):
        orig_val_loader = DataLoader(
            self.orig_val_dataset,
            shuffle=False,
            drop_last=False,
            batch_size=self.cfg.train.val_bsz,
            num_workers=self.cfg.num_workers,
            pin_memory=self.use_cuda,
        )

        uni_val_loader = DataLoader(
            self.uni_val_dataset,
            shuffle=False,
            drop_last=False,
            batch_size=self.cfg.train.val_bsz,
            num_workers=self.cfg.num_workers,
            pin_memory=self.use_cuda,
        )

        return [orig_val_loader, uni_val_loader]

    def validation_step(self, batch, batch_nb, dataset_idx=None):
        tgt_img, tgt_midreps = batch
        if "autoencoder" in self.cfg.tasks:
            tgt_midreps["autoencoder"] = tgt_img
        _, pred_midreps = self(tgt_img)
        if dataset_idx == 0:
            prefix = "orig"
        elif dataset_idx == 1:
            prefix = "uni"
        else:
            prefix = ""
        stats = {
            "lnorm": {},
            "acc": {},
        }
        for task in self.cfg.tasks:
            lnorm = self.lnorm(pred_midreps[task], tgt_midreps[task]).flatten(1).sum(1)
            abs_dist = (pred_midreps[task] - tgt_midreps[task]).abs()
            abs_dist = abs_dist.mean(1).flatten(1)

            stats["lnorm"][task] = lnorm

            acc_levels = [0.01, 0.025, 0.05]
            stats["acc"][task] = {}
            for acc_l in acc_levels:
                stats["acc"][task][acc_l] = get_acc_at(abs_dist, acc_l)

        if batch_nb == 0:
            num_samples = min(tgt_img.size(0), 32)
            if ((0 > tgt_img).any() or (tgt_img > 1).any()):
                print('Tgt img out of range')
            img1 = make_grid(tgt_img[:num_samples], nrow=1).permute(1, 2, 0).cpu()
            img2 = make_grid(tgt_midreps["depths"][:num_samples], nrow=1)[0].cpu()
            img3 = make_grid(pred_midreps["depths"][:num_samples], nrow=1)[0].cpu()

            fig, axes = plt.subplots(
                nrows=1,
                ncols=1 + 2 * len(pred_midreps),
                figsize=(6 * (1 + 2 * len(pred_midreps)), 6 * num_samples),
            )

            img_ax = axes[0]
            img = make_grid(tgt_img[:num_samples], nrow=1).permute(1, 2, 0).cpu()
            img_ax.imshow(img)
            img_ax.set_title("Input image", fontsize=30)
            img_ax.axis("off")

            for i, task_name in enumerate(pred_midreps.keys()):
                gt_ax = axes[i * 2 + 1]
                pred_ax = axes[i * 2 + 2]

                gt_img = make_grid(tgt_midreps[task_name][:num_samples], nrow=1).cpu()
                if gt_img.size(0) == 1:
                    gt_img = gt_img.squeeze(0)
                else:
                    gt_img = gt_img.permute(1, 2, 0)
                pred_img = make_grid(
                    pred_midreps[task_name][:num_samples], nrow=1
                ).cpu()
                if pred_img.size(0) == 1:
                    pred_img = pred_img.squeeze(0)
                else:
                    pred_img = pred_img.permute(1, 2, 0)

                if ((0 > gt_img).any() or (gt_img > 1).any()):
                    print(f"gt img {task_name} out of range")
                if ((0 > pred_img).any() or (pred_img > 1).any()):
                    print(f"pred img {task_name} out of range")

                gt_ax.imshow(gt_img, cmap="viridis")
                pred_ax.imshow(pred_img, cmap="viridis")

                gt_ax.set_title(f"Real {task_name}", fontsize=30)
                pred_ax.set_title(f"Predicted {task_name}", fontsize=30)
                gt_ax.axis("off")
                pred_ax.axis("off")

            plt.subplots_adjust(wspace=0.01)
            plt.axis("off")
            self.log_figure(fig, f"{prefix}_validation", self.global_step)

        return stats

    def validation_epoch_end(self, outputs):
        # stats = {"orig": {"lnorm": {}, "acc": {}}, "uni": {"lnorm": {}, "acc": {}}}
        stats = {}

        def aggregate_lnorm(output_list, prefix):
            for task_name in output_list[0]["lnorm"].keys():
                # stats[prefix]["lnorm"][task_name] = torch.cat(
                stats[f"val_{prefix}_lnorm_{task_name}"] = (
                    torch.cat([o["lnorm"][task_name] for o in output_list])
                    .mean()
                    .item()
                )

        def aggregate_acc(output_list, prefix):
            for task_name in output_list[0]["acc"].keys():
                # stats[prefix]["acc"][task_name] = {}
                for acc_l in output_list[0]["acc"][task_name].keys():
                    # stats[prefix]["acc"][task_name][acc_l] = torch.cat(
                    stats[f"val_{prefix}_acc_{task_name}_{acc_l}"] = (
                        torch.cat([o["acc"][task_name][acc_l] for o in output_list])
                        .mean()
                        .item()
                    )

        aggregate_lnorm(outputs[0], "orig")
        aggregate_lnorm(outputs[1], "uni")
        aggregate_acc(outputs[0], "orig")
        aggregate_acc(outputs[1], "uni")

        stats = flatten_json_iterative_solution(stats)
        stats["Epoch"] = self.current_epoch

        self.print()
        self.print(f"Epoch: {self.current_epoch}")
        self.print(pp.pformat(stats))

        return {
            # "progress_bar": {
            #     "val_depths_lnorm": stats["val_depths_lnorm"],
            #     "val_depths_acc_0.1": stats["val_depths_acc_0.1"],
            # },
            "log": stats,
            # "val_depths_acc_0.025": stats["orig"]["acc"],
        }

    def training_step(self, batch, batch_nb, optimizer_idx=None):
        tgt_img, tgt_midreps = batch
        if "autoencoder" in self.cfg.tasks:
            tgt_midreps["autoencoder"] = tgt_img
        if (optimizer_idx == 0) or (optimizer_idx is None):
            # Train encoder decoder

            _, pred_midreps = self(tgt_img)
            losses = {}
            stats = {}
            for task_name in self.cfg.tasks:
                if self.discriminators:
                    disc = self.discriminators[task_name]
                    real_inp = torch.cat([tgt_img, tgt_midreps[task_name]], 1)
                    real_feats = disc.first_conv(real_inp)

                    fake_inp = torch.cat([tgt_img, pred_midreps[task_name]], 1)
                    fake_feats = disc.first_conv(fake_inp)

                    lnorm_loss = self.lnorm(fake_feats, real_feats)
                    lnorm_loss = lnorm_loss.flatten(1).sum(1).mean()
                else:
                    lnorm_loss = self.lnorm(pred_midreps[task_name], tgt_midreps[task_name])
                    lnorm_loss = lnorm_loss.flatten(1).sum(1).mean() * self.lambdas.lnorm
                
                abs_dist = (pred_midreps[task_name] - tgt_midreps[task_name]).abs()
                abs_dist = abs_dist.mean(1).flatten(1)
                losses[task] = {
                    "loss": lnorm_loss,
                    "lnorm": lnorm_loss,
                }
                acc_levels = [0.01, 0.025, 0.05]
                stats[task] = {
                    f"acc_{acc_l}": get_acc_at(abs_dist, acc_l).mean()
                    for acc_l in acc_levels
                }

            # if self.discriminators:
            #     for task_name, d in self.discriminators.items():
            #         disc_inp = torch.cat([tgt_img, pred_midreps[task_name]], 1)
            #         disc_inp += torch.empty_like(disc_inp).normal_(0, 0.01)
            #         disc_pred = d(disc_inp)
            #         gen_loss = bce_fill(disc_pred, 1).mean()
            #         losses[task]["g"] = gen_loss
            #         if gen_loss >= 0.05 and getattr(self, "d_loss", 0.0) <= 1.5:
            #             losses[task]["loss"] += gen_loss * self.lambdas.adv

            total_loss = 0.0
            for task, loss in losses.items():
                total_loss = total_loss + loss["loss"] * self.lambdas[task]

            tqdm_dict = {}
            for task, loss in losses.items():
                for lname, lval in loss.items():
                    if lname != "loss":
                        tqdm_dict[f"{task}_{lname}"] = lval
                tqdm_dict[f"{task}_acc_0.01"] = stats[task]["acc_0.01"]

            log = {**tqdm_dict}
            for task, _stats in stats.items():
                for sname, _stat in _stats.items():
                    log[f"{task}_{sname}"] = _stat

            if optimizer_idx == 0:
                self.stored_fake_batch = edict(
                    **{name: mr.detach() for name, mr in pred_midreps.items()}
                )

        elif optimizer_idx == 1:
            pred_midreps = self.stored_fake_batch
            losses = {}
            for task_name, disc in self.discriminators.items():
                real_inp = torch.cat([tgt_img, tgt_midreps[task_name]], 1)
                # real_inp += torch.empty_like(real_inp).normal_(0, 0.01)
                _, real2real = disc(real_inp)

                fake_inp = torch.cat([tgt_img, pred_midreps[task_name]], 1)
                # fake_inp += torch.empty_like(fake_inp).normal_(0, 0.01)
                _, fake2real = disc(fake_inp)

                disc_loss = (
                    bce_fill(real2real, 1).mean() + bce_fill(fake2real, 0).mean()
                )
                disc_loss = disc_loss * 0.5
                losses[task_name] = disc_loss

            total_loss = torch.tensor(0.0, requires_grad=True, device=tgt_img.device)
            for task, loss in losses.items():
                # if loss >= 0.2:
                total_loss = total_loss + loss * self.lambdas[task]

            self.d_loss = total_loss.detach()

            tqdm_dict = {}
            for task, loss in losses.items():
                tqdm_dict[f"{task}_d"] = loss

            log = tqdm_dict

        return {
            "loss": total_loss,
            "progress_bar": tqdm_dict,
            "log": log,
        }
