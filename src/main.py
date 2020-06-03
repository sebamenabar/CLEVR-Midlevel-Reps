import os
import os.path as osp
from pprint import PrettyPrinter as PP

from dotenv import load_dotenv
from easydict import EasyDict as edict

import torch
import torch.nn as nn
import pytorch_lightning as pl
# from model import PLModel
from simple_model import PLModel
from base_pl_model import CustomProgressBar
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
    loggers = model.make_lightning_loggers()

    default_ckpt_callback_kwargs = {
        "filepath": osp.join(model.exp_dir, "checkpoints/"),
        # "monitor": "val_depths_acc_0.025",
        "verbose": True,
        "save_top_k": -1,
    }
    ckpt_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        **default_ckpt_callback_kwargs,
    )

    # progress_bar = CustomProgressBar(
    #     refresh_rate=args.progress_bar_refresh_rate,
    #     process_position=args.process_position,
    # )
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=loggers,
        # callbacks=[progress_bar],
        max_epochs=cfg.train.epochs,
        default_root_dir=model.exp_dir,
        checkpoint_callback=ckpt_callback,
    )
    if args.eval:
        pass
    elif args.test:
        pass
    else:
        pass
        trainer.fit(model)
