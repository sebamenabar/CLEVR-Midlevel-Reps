from easydict import EasyDict as edict
from collections import OrderedDict as odict

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision.transforms as T

from data import CLEVRMidrepsDataset
from base_pl_model import BasePLModel
from discriminator import NLayerDiscriminator
from backbones import (
    MRBackbone,
    Midreps,
    MRDecoder,
    RNBackbone,
    RNDecoder,
)
from common import task_to_out_nc
from losses import bce_fill



class PLModel(BasePLModel):
    def __init__(self, cfg=None):
        super().__init__(cfg)

        if cfg.model.arch == "rn":
            self.backbone = RNBackbone()
            self.decoder = RNDecoder(out_dim=len(cfg.tasks))

        # self.backbone = MRBackbone(**cfg.model.backbone.kwargs)
        # self.midreps = None
        # self.decoders = None
        # self.discriminators = None

        # decoder_in_nc = 2048
        # if cfg.model.midreps.use:
        #     midreps = {}
        #     decoder_in_nc = 8
        #     for task in cfg.tasks:
        #         midreps[task] = Midreps(**cfg.model.midreps.kwargs)
        #     self.midreps = nn.Sequential(odict(midreps))

        # cfg.model.decoder.kwargs.in_nc = decoder_in_nc
        # decoders = {}
        # if cfg.model.discriminator.use:
        #     discriminators = {}
        # for task in cfg.tasks:
        #     out_nc = task_to_out_nc[task]
        #     decoder_kwargs = {}
        #     decoder_kwargs.update(cfg.model.decoder.kwargs)
        #     decoder_kwargs["out_channels"] = out_nc
        #     decoders[task] = MRDecoder(**decoder_kwargs)
        #     if cfg.model.discriminator.use:
        #         disc_kwargs = {}
        #         disc_kwargs.update(cfg.model.discriminator.kwargs)
        #         disc_kwargs["input_nc"] = 3 + out_nc
        #         discriminators[task] = NLayerDiscriminator(**disc_kwargs)
        # self.decoders = nn.Sequential(odict(decoders))
        # if cfg.model.discriminator.use:
        #     self.discriminators = nn.Sequential(odict(discriminators))

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

        self.print("Multitask weights:")
        self.print(self.lambdas)

    def forward(self, img):
        features = self.backbone(img)
        decoded = self.decoder(features)
        # ret = edict()
        # if self.midreps:
        #     for task_name, task_net in self.midreps._modules.items():
        #         ret[task_name] = task_net(features)
        #     for task_name, task_decoder in self.decoders._modules.items():
        #         ret[task_name] = task_decoder(ret[task_name])
        # else:
        #     for task_name, task_decoder in self.decoders._modules.items():
        #         ret[task_name] = task_decoder(features)

        return features, decoded

    def configure_optimizers(self):
        encoder_decoder_params = list(self.backbone.parameters())
        encoder_decoder_params += list(self.decoder.parameters())
        # if self.midreps:
        #     encoder_decoders_params += list(self.midreps.parameters())
        # if self.decoders:
        #     encoder_decoders_params += list(self.decoders.parameters())

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
            encoder_decoder_opt, milestones=[5, 10], gamma=0.1,
        )
        if self.discriminators:
            discriminator_opt = make_opt(
                self.discriminators.parameters(), "discriminator"
            )
            return [encoder_decoder_opt, discriminator_opt], [encoder_decoder_sch]

        return [encoder_decoder_opt], [encoder_decoder_sch]

    def prepare_data(self):
        self.train_dataset = CLEVRMidrepsDataset(
            base_dir=self.cfg.data_dir,
            split="train",
            midreps=self.cfg.tasks,
            transform=CLEVRMidrepsDataset.std_img_transform,
            midreps_transform=CLEVRMidrepsDataset.std_midreps_transforms,
        )

        self.val_dataset = CLEVRMidrepsDataset(
            base_dir=self.cfg.data_dir,
            split="val",
            midreps=self.cfg.tasks,
            transform=CLEVRMidrepsDataset.std_img_transform,
            midreps_transform=CLEVRMidrepsDataset.std_midreps_transforms,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            batch_size=self.cfg.train.bsz,
            num_workers=self.cfg.num_workers,
            pin_memory=self.use_cuda,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            drop_last=False,
            batch_size=self.cfg.train.val_bsz,
            num_workers=self.cfg.num_workers,
            pin_memory=self.use_cuda,
        )

    def validation_step(self, batch, batch_nb):
        tgt_img, tgt_midreps = batch
        _, pred_midreps = self(tgt_img)
        stats = {}
        # progress_bar = {}
        for task in self.cfg.tasks:
            lnorm = self.lnorm(pred_midreps[task], tgt_midreps[task]).flatten(1).sum(1)
            abs_dist = (pred_midreps[task] - tgt_midreps[task]).abs()
            abs_dist = abs_dist.mean(1).flatten(1)
            acc_at_0025 = (
                (abs_dist <= 0.025).sum(1).float() / abs_dist.size(1)
            ).float()
            acc_at_005 = ((abs_dist <= 0.05).sum(1).float() / abs_dist.size(1)).float()
            acc_at_01 = ((abs_dist <= 0.1).sum(1).float() / abs_dist.size(1)).float()

            # progress_bar[f"{task}_lnorm"] = lnorm.mean()
            # progress_bar[f"{task}_acc_0.025"] = acc_at_0025.mean()

            stats[f"val_{task}_lnorm"] = lnorm
            stats[f"val_{task}_acc_0.025"] = acc_at_0025
            stats[f"val_{task}_acc_0.05"] = acc_at_005
            stats[f"val_{task}_acc_0.1"] = acc_at_01

        if batch_nb == 0:
            img1 = make_grid(tgt_img[:32], nrow=1).permute(1, 2, 0).cpu()
            img2 = make_grid(tgt_midreps["depths"][:32], nrow=1)[0].cpu()
            img3 = make_grid(pred_midreps["depths"][:32], nrow=1)[0].cpu()

            fig, (ax1, ax2, ax3) = plt.subplots(
                nrows=1, ncols=3, figsize=(24, 8 * tgt_img.size(0))
            )
            ax1.imshow(img1)
            ax2.imshow(img2, cmap="viridis")
            ax3.imshow(img3, cmap="viridis")

            ax1.set_title("Input image")
            ax2.set_title("Real depth")
            ax3.set_title("Predicted depth")

            ax1.axis("off")
            ax2.axis("off")
            ax3.axis("off")

            plt.subplots_adjust(wspace=0.01)

            self.log_figure(fig, "depths", self.global_step)

        return stats

    def validation_epoch_end(self, outputs):
        stats = {}
        for k in outputs[0].keys():
            stats[k] = torch.cat([o[k] for o in outputs]).mean()

        return {
            "progress_bar": {
                "val_depths_lnorm": stats["val_depths_lnorm"],
                "val_depths_acc_0.1": stats["val_depths_acc_0.1"],
            },
            "log": stats,
        }

    def training_step(self, batch, batch_nb, optimizer_idx=None):
        if (optimizer_idx == 0) or (optimizer_idx is None):
            # Train encoder decoder
            tgt_img, tgt_midreps = batch
            _, pred_midreps = self(tgt_img)
            losses = {}
            stats = {}
            for task in self.cfg.tasks:
                lnorm_loss = self.lnorm(pred_midreps[task], tgt_midreps[task])
                lnorm_loss = lnorm_loss.flatten(1).sum(1).mean() * self.lambdas.lnorm
                abs_dist = (pred_midreps[task] - tgt_midreps[task]).abs()
                abs_dist = abs_dist.mean(1).flatten(1)
                acc_at_0025 = (
                    ((abs_dist <= 0.025).sum(1).float() / abs_dist.size(1))
                    .float()
                    .mean()
                )
                acc_at_005 = (
                    ((abs_dist <= 0.05).sum(1).float() / abs_dist.size(1))
                    .float()
                    .mean()
                )
                acc_at_01 = (
                    ((abs_dist <= 0.1).sum(1).float() / abs_dist.size(1)).float().mean()
                )
                losses[task] = {
                    "loss": lnorm_loss,
                    "lnorm": lnorm_loss,
                }
                stats[task] = {
                    "acc_0.025": acc_at_0025,
                    "acc_0.05": acc_at_005,
                    "acc_0.1": acc_at_01,
                }
                if self.discriminators:
                    disc_inp = torch.cat([tgt_img, pred_midreps[task]], 1)
                    disc_inp += torch.empty_like(disc_inp).normal_(0, 0.01)
                    disc_pred = self.discriminators._modules[task](disc_inp)
                    gen_loss = bce_fill(disc_pred, 1)
                    gen_loss = gen_loss.mean() * self.lambdas.adv
                    losses[task]["loss"] += gen_loss
                    losses[task]["g"] = gen_loss

            total_loss = 0.0
            for task, loss in losses.items():
                total_loss += loss["loss"] * self.lambdas[task]

            tqdm_dict = {}
            for task, loss in losses.items():
                for lname, lval in loss.items():
                    if lname != "loss":
                        tqdm_dict[f"{task}_{lname}"] = lval
                tqdm_dict[f"{task}_acc_0.1"] = stats[task]["acc_0.1"]

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
            tgt_img, tgt_midreps = batch
            losses = {}
            for task_name, disc in self.discriminators._modules.items():
                real_inp = torch.cat([tgt_img, tgt_midreps[task_name]], 1)
                real_inp += torch.empty_like(real_inp).normal_(0, 0.01)
                real2real = disc(real_inp)

                fake_inp = torch.cat([tgt_img, pred_midreps[task_name]], 1)
                fake_inp += torch.empty_like(fake_inp).normal_(0, 0.01)
                fake2real = disc(fake_inp)

                disc_loss = (
                    bce_fill(real2real, 1).mean() + bce_fill(fake2real, 0).mean()
                )
                disc_loss = disc_loss * 0.5 * self.lambdas.adv
                losses[task_name] = disc_loss

            total_loss = 0.0
            for task, loss in losses.items():
                total_loss += loss * self.lambdas[task]

            tqdm_dict = {}
            for task, loss in losses.items():
                tqdm_dict[f"{task}_d"] = loss

            log = tqdm_dict

        return {
            "loss": total_loss,
            "progress_bar": tqdm_dict,
            "log": log,
        }
