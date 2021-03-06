import os
import os.path as osp
import sys
import yaml
import json
import shutil
import random
from dateutil import tz
from datetime import datetime as dt

from tqdm.auto import tqdm
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only

from base_config import mkdir_p

_plt = None


class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        sys.stdout.flush()
        self.log.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class BasePLModel(pl.LightningModule):
    def __init__(self, hparams=None):
        super().__init__()

        # self.net = Net()
        self.hparams = hparams
        self.cfg = hparams
        # if hparams.resume:
        #     self.load_from_checkpoint(hparams.resume)
        self.cfg = hparams
        self.comet_set = False
        self._run_name = None
        self.work_dir = osp.dirname(osp.dirname(osp.realpath(__file__)))
        self.exp_dir = ""

    @property
    def use_cuda(self):
        return self.cfg.gpus is not None and self.cfg.gpus != 0

    def on_epoch_start(self):
        if not self.comet_set:
            if self.cfg.logcomet:
                self.set_comet_graph()
                self.log_src_comet()
                self.log_cfg_comet()
                # self.print("\nSent src and config to comet")
            self.comet_set = True  # Avoiding calling each time epoch starts

        return super().on_epoch_start()

    @property
    def loggers(self):
        if self.logger:
            if isinstance(self.logger, pl.loggers.base.LoggerCollection):
                return self.logger
            else:
                return [self.logger]

    @property
    def comet_logger(self):
        comet_logger = getattr(self, "_comet_logger", None)
        if comet_logger:
            return comet_logger
        for logger in self.loggers:
            if isinstance(logger, pl.loggers.CometLogger):
                self._comet_logger = logger
                return logger

    @rank_zero_only
    def set_comet_graph(self):
        if self.cfg.logcomet:
            self.comet_logger.experiment.set_model_graph(str(self))

    @rank_zero_only
    def log_src_comet(self):
        if self.cfg.logcomet:
            self.comet_logger.experiment.log_asset_folder(
                osp.join(self.work_dir, "src"), recursive=True, log_file_name=True
            )
            print("Sent src to comet")
            

    @rank_zero_only
    def log_cfg_comet(self):
        if self.cfg.logcomet and len(self.exp_dir):
            self.comet_logger.experiment.log_asset(osp.join(self.exp_dir, "cfg.json"))
            self.comet_logger.experiment.log_asset(osp.join(self.exp_dir, "cfg.yml"))
            if os.path.exists(osp.join(self.exp_dir, "args.json")):
                self.comet_logger.experiment.log_asset(
                    osp.join(self.exp_dir, "args.json")
                )
            print("Sent config to comet")

    @rank_zero_only
    def log_figure(self, figure, name, step=None, close=True):
        if self.logger:
            for logger in self.loggers:
                if type(logger) is pl.loggers.TensorBoardLogger:
                    logger.experiment.add_figure(
                        name, figure, global_step=step, close=False
                    )
                elif type(logger) is pl.loggers.CometLogger:
                    logger.experiment.log_figure(name, figure, step=step)

        if close:
            global _plt
            if _plt is None:
                import matplotlib.pyplot as plt

                _plt = plt
            plt = _plt
            plt.close(figure)

    @property
    def run_name(self):
        if self._run_name:
            return self._run_name
        return ""

    def make_comet_logger(self):
        return pl.loggers.CometLogger(
            api_key=os.environ["COMET_API_KEY"],
            workspace=self.cfg.comet_workspace,
            project_name=self.cfg.comet_project_name,
            experiment_name=self.run_name,
        )

    # def make_lightning_loggers_ckpt(
    def make_lightning_loggers(
        self,
        # ckpt_callback_kwargs=dict(
        #     # filepath=osp.join(self.exp_dir, "checkpoints/"),
        #     monitor="val_loss",
        #     verbose=True,
        #     save_top_k=2,
        # ),
    ):
        loggers = [pl.loggers.TensorBoardLogger(save_dir=self.exp_dir, name="",)]
        if self.cfg.logcomet:
            loggers.append(self.make_comet_logger())

        # default_ckpt_callback_kwargs = {
        #     "filepath": osp.join(self.exp_dir, "checkpoints/"),
        # }
        # default_ckpt_callback_kwargs.update(ckpt_callback_kwargs)
        # ckpt_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        #     **default_ckpt_callback_kwargs,
        # )
        return loggers  # , ckpt_callback

    @rank_zero_only
    def init_log(self, args=None):
        now = dt.now(tz.tzlocal())
        now = now.strftime("%m-%d-%Y-%H-%M-%S")
        log_dir = self.cfg.exp_name
        run_name = self.cfg.run_name
        if run_name == "" or run_name is None:
            run_name = now
        self._run_name = run_name
        exp_dir = osp.join(self.work_dir, "experiments", log_dir, run_name)
        _exp_dir = exp_dir
        i = 2
        while os.path.exists(_exp_dir):
            _exp_dir = f"{exp_dir}-{i}"
            i += 1
        exp_dir = _exp_dir

        self.exp_dir = exp_dir
        shutil.copytree(
            osp.join(self.work_dir, "src"),
            osp.join(self.exp_dir, "src"),
            ignore=shutil.ignore_patterns(".*", "__pycache__", ".DS_Store"),
        )

        self.logfile = osp.join(exp_dir, "logfile.log")
        sys.stdout = Logger(self.logfile)

        with open(osp.join(self.exp_dir, "cfg.json"), "w") as f:
            json.dump(self.cfg, f, indent=4)
        with open(osp.join(self.exp_dir, "cfg.yml"), "w") as f:
            yaml.dump(json.loads(json.dumps(self.cfg)), f, indent=4)
        if args is not None:
            with open(osp.join(self.exp_dir, "args.json"), "w") as f:
                json.dump(args, f, indent=4)

        # print(exp_dir)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class CustomProgressBar(pl.callbacks.progress.ProgressBar):
    # Can't get rid of the v_num
    # def on_batch_end(self, trainer, pl_module):
    #     super().on_batch_end(trainer, pl_module)
    #     if self.is_enabled and self.train_batch_idx % self.refresh_rate == 0:
    #         self.main_progress_bar.update(self.refresh_rate)
    #         progress_bar_dict = trainer.progress_bar_dict
    #         del progress_bar_dict["v_num"]
    #         self.main_progress_bar.set_postfix(progress_bar_dict)

    def init_sanity_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for the validation sanity run. """
        bar = tqdm(
            desc="Validation sanity check",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stderr,
        )
        return bar

    def init_train_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for training. """
        bar = tqdm(
            desc="Training",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stderr,
            smoothing=0,
        )
        return bar

    def init_validation_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for validation. """
        bar = tqdm(
            desc="Validating",
            position=(2 * self.process_position + 1),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stderr,
        )
        return bar

    def init_test_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for testing. """
        bar = tqdm(
            desc="Testing",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stderr,
        )
        return bar
