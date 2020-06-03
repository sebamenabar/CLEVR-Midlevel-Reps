import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from backbones import RNEncoder, RNDecoder, RNDiscriminator
from base_pl_model import BasePLModel
from utils import task_to_out_nc, get_acc_at
from losses import bce_fill


class PLModel(BasePLModel):
    def __init__(self, cfg=None):
        super().__init__(cfg)

        self.many_validation = True
        self.encoder = RNEncoder(**cfg.model.encoder.kwargs)
        self.decoder = RNDecoder(tasks=cfg.tasks, **cfg.model.decoder.kwargs)

        self.discriminators = None
        if cfg.discriminator.use:
            self.discriminators = nn.ModuleDict()
            for task in cfg.tasks:
                self.discriminator[task] = RNDiscriminator(
                    input_nc=3 + task_to_out_nc[task],
                    **cfg.model.discriminator.kwargs,
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
            encoder_decoder_opt, milestones=[5, 10], gamma=0.1,
        )
        if self.discriminators:
            discriminator_opt = make_opt(
                self.discriminators.parameters(), "discriminator"
            )
            return [encoder_decoder_opt, discriminator_opt], [encoder_decoder_sch]

        return [encoder_decoder_opt], [encoder_decoder_sch]

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
        stats = {}
        if dataset_idx == 0:
            prefix = "orig_"
        elif dataset_idx == 1:
            prefix = "uni_"
        else:
            prefix = ""
        for task in self.cfg.tasks:
            lnorm = self.lnorm(pred_midreps[task], tgt_midreps[task]).flatten(1).sum(1)
            abs_dist = (pred_midreps[task] - tgt_midreps[task]).abs()
            abs_dist = abs_dist.mean(1).flatten(1)

            stats[f"{prefix}val_{task}_lnorm"] = lnorm
            acc_levels = [0.01, 0.025, 0.05]
            for acc_l in acc_levels:
                stats[f"{prefix}val_{task}_{acc_l}"] = get_acc_at(abs_dist, acc_l)

        if batch_nb == 0:
            num_samples = 32
            img1 = make_grid(tgt_img[:num_samples], nrow=1).permute(1, 2, 0).cpu()
            img2 = make_grid(tgt_midreps["depths"][:num_samples], nrow=1)[0].cpu()
            img3 = make_grid(pred_midreps["depths"][:num_samples], nrow=1)[0].cpu()

            fig, axes = plt.subplots(
                nrows=1,
                ncols=1 + 2 * len(pred_midreps),
                figsize=(6 * (1 + 2 * len(pred_midreps)), 6 * num_samples),
            )
            plt.axis('off')
            
            img_ax = axes[0]
            img = make_grid(tgt_img[:num_samples], nrow=1).permute(1, 2, 0).cpu()
            img_ax.imshow(img)
            img_ax.set_title("Input image", fontsize=30)
            plt.subplots_adjust(wspace=0.01)
            # img_ax.axis("off")
            
            for i, task_name in enumerate(pred_midreps.keys()):
                gt_ax = axes[i * 2 + 1]
                pred_ax = axes[i * 2 + 2]
                
                gt_img = make_grid(tgt_midreps[task_name][:num_samples], nrow=1).cpu()
                if gt_img.size(0) == 1:
                    gt_img = gt_img.squeeze(0)
                else:
                    gt_img = gt_img.permute(1, 2, 0)
                pred_img = make_grid(pred_midreps[task_name][:num_samples], nrow=1).cpu()
                if pred_img.size(0) == 1:
                    pred_img = pred_img.squeeze(0)
                else:
                    pred_img = pred_img.permute(1, 2, 0)
            
                gt_ax.imshow(gt_img, cmap="viridis")
                pred_ax.imshow(pred_img, cmap="viridis")

                gt_ax.set_title(f"Real {task_name}", fontsize=30)
                pred_ax.set_title(f"Predicted {task_name}", fontsize=30)
            
            self.log_figure(fig, f"{prefix}validation", self.global_step)

        return stats

    def validation_epoch_end(self, outputs):
        stats = {}
        outputs = outputs[0]
        for k in outputs[0].keys():
            stats[k] = torch.cat([o[k] for o in outputs]).mean()

        return {
#             "progress_bar": {
#                 "val_depths_lnorm": stats["val_depths_lnorm"],
#                 "val_depths_acc_0.1": stats["val_depths_acc_0.1"],
#             },
            "log": stats,
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
            for task in self.cfg.tasks:
                lnorm_loss = self.lnorm(pred_midreps[task], tgt_midreps[task])
                lnorm_loss = lnorm_loss.flatten(1).sum(1).mean() * self.lambdas.lnorm
                abs_dist = (pred_midreps[task] - tgt_midreps[task]).abs()
                abs_dist = abs_dist.mean(1).flatten(1)
                losses[task] = {
                    "loss": lnorm_loss,
                    "lnorm": lnorm_loss,
                }
                acc_levels = [0.01, 0.025, 0.05]
                stats[task] = {
                    f"acc_{acc_l}": get_acc_at(abs_dist, acc_l).mean() for acc_l in acc_levels
                }

            total_loss = 0.0
            for task, loss in losses.items():
                total_loss += loss["loss"] * self.lambdas[task]

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