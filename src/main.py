import os
from pprint import PrettyPrinter as PP

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
    pp = PP(indent=4)

    model = PLModel(cfg)
    # Prints should be done after the init log
    model.init_log(vars(args))
    pp.pprint(vars(args))
    pp.pprint(cfg)
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
