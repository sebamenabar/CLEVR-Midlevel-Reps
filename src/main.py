from easydict import EasyDict as edict
from dotenv import load_dotenv

import torch
import torch.nn as nn
import pytorch_lightning as pl
from base_pl_model import BasePLModel
from config import __C, parse_args_and_set_config

### EXAMPLE SPECIFIC
import os
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

load_dotenv()

__C.model = edict(
    in_dim=(784, edict(type=int)),
    hidden_dim=1000,
    out_dim=10,
)


if __name__ == "__main__":
    args, cfg = parse_args_and_set_config(__C)
    print(vars(args))
    print(cfg)

    # model = Model(cfg)
    # model.init_log(vars(args))
    # loggers, ckpt_callback = model.make_lightning_loggers_ckpt()
    # trainer = pl.Trainer.from_argparse_args(
    #     args,
    #     logger=loggers,
    #     checkpoint_callback=ckpt_callback,
    #     max_epochs=cfg.train.epochs,
    #     default_root_dir=model.exp_dir,
    # )
    # if args.eval:
    #     pass
    # elif args.test:
    #     pass
    # else:
    #     pass
    #     trainer.fit(model)
