import os

from dotenv import load_dotenv
from easydict import EasyDict as edict

import torch
import torch.nn as nn
import pytorch_lightning as pl
from model import PLModel
from config import __C, parse_args_and_set_config


load_dotenv()


if __name__ == "__main__":
    args, cfg = parse_args_and_set_config(__C)

    model = PLModel(cfg)
    model.init_log(vars(args))
    loggers, ckpt_callback = model.make_lightning_loggers_ckpt()
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=loggers,
        checkpoint_callback=ckpt_callback,
        max_epochs=cfg.train.epochs,
        default_root_dir=model.exp_dir,
    )
    if args.eval:
        pass
    elif args.test:
        pass
    else:
        pass
        trainer.fit(model)
