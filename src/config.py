import torch
from pytorch_lightning.utilities import parsing
from base_config import __C, parse_args_and_set_config, edict, _to_values_only

# Remove unrequired parameters
del __C.train.lr

# Replace default arguments
__C.tasks = ("depths", edict(choices=["depths", "normals", "autoencoder"], nargs="+", type=str))
if torch.cuda.is_available():
    __C.orig_dir = (
        "/storage1/samenabar/code/CLMAC/clevr-dataset-gen/datasets/CLEVR_v1.2",
        edict(type=str),
    )
    __C.uni_dir = (
        "/storage1/samenabar/code/CLMAC/clevr-dataset-gen/datasets/CLEVR_Uni_v1.2",
        edict(type=str),
    )
else:
    __C.orig_dir = (
        "/Users/sebamenabar/Documents/datasets/tmp/CLEVR_v1.2",
        edict(type=str),
    )
    __C.uni_dir = (
        "/Users/sebamenabar/Documents/datasets/tmp/CLEVR_Uni_v1.2",
        edict(type=str),
    )
__C.train.epochs[0] = 15
__C.train.bsz[0] = 64
__C.train.val_bsz[0] = 64
__C.train.lrs = edict()
__C.train.lrs.default = (1e-4, edict(type=float))
__C.train.lrs.discriminator = (1e-5, edict(type=float))
if torch.cuda.is_available():
    __C.num_workers[0] = 8
else:
    __C.num_workers[0] = 4

# Add custom arguments
__C.train.lnorm = ("smooth_l1", edict(choices=["l1", "l2", "smooth_l1"], type=str))
__C.train.weight_decay = edict()
__C.train.weight_decay.default = 2e-6
__C.train.weight_decay.discriminator = 10 * __C.train.weight_decay.default
__C.train.lnorm_mult = (0.996, edict(type=float))
__C.train.adv_mult = (0.004, edict(type=float))
__C.train.task_lambdas = edict()
__C.train.task_lambdas.depths = 1
__C.train.task_lambdas.normals = 1
__C.train.task_lambdas.autoencoder = 1
__C.train.adv_skip = (True, edict(type=lambda x: bool(parsing.strtobool(x))))

# Model args
__C.model = edict()
__C.model.arch = ("rn", edict(type=str, choices=["rn"]))

# Encoder args
# __C.model.backbone = edict()
#  __C.model.backbone.use = True
# __C.model.backbone.kwargs = edict(lightweight=True, layers=None)
__C.model.encoder = edict(kwargs=edict(out_nc=512))


# Midreps args
# __C.model.midreps = edict()
# __C.model.midreps.use = (
#     True,
#     edict(type=lambda x: bool(parsing.strtobool(x))),
# )
# __C.model.midreps.kwargs = edict(normalize_outputs=True)

# Decoder args
__C.model.decoder = edict()
# __C.model.decoder.use = True
__C.model.decoder.kwargs = edict(
    # in_nc=8, output_act="sigmoid", out_channels=3, lightweight=True,
    in_nc=__C.model.encoder.kwargs.out_nc,
    last_nc=256,
)

# Discriminator args
__C.model.discriminator = edict()
__C.model.discriminator.use = (
    False,
    edict(type=lambda x: bool(parsing.strtobool(x)),),
)
__C.model.discriminator.kwargs = edict(
    out_nc=256,

    # input_nc=6,
    # ndf=64,
    # n_layers=5,
    # norm_layer="batchnorm",
    # stride=2,
    # use_sigmoid=False,
    # out_pool=True,
)
