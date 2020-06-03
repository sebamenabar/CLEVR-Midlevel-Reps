import torch.nn as nn
from utils import task_to_out_nc


out_activations = {
    "depths": nn.functional.relu,
    "normals": nn.functional.relu,
    "autoencoder": nn.functional.relu,
}

class RNBackbone(nn.Module):
    def __init__(self, out_nc=512):
        super().__init__()
        nc = out_nc // (2 ** 3)
        layers = [
            nn.Conv2d(3, nc, [3, 3], 2, 1, bias=False),
            nn.BatchNorm2d(nc),
            nn.ReLU(),
        ]
        for i in range(1, 4):
            layers += [
                nn.Conv2d(nc, nc * 2, [3, 3], 2, 1, bias=False),
                nn.BatchNorm2d(nc * 2),
                nn.ReLU(),
            ]
            nc = 2 * nc
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class RNDiscriminator(nn.Module):
    def __init__(self, in_nc=4, out_nc=512):
        super().__init__()
        nc = out_nc // (2 ** 5)
        layers = [
            nn.Conv2d(in_nc, nc, 4, 2, 0, bias=False),
            nn.BatchNorm2d(nc),
            nn.LeakyReLU(0.2),
        ]
        for i in range(1, 4):
            layers += [
                nn.Conv2d(nc, nc * 2, 4, 2, 0, bias=False),
                nn.BatchNorm2d(nc * 2),
                nn.LeakyReLU(0.2)
            ]
            nc = 2 * nc

        layers += [
            nn.Conv2d(nc, nc * 2, 5, 2, 0, bias=False),
            nn.BatchNorm2d(nc * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nc * 2, nc * 4, 4, 1, 0, bias=False)
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x is in range [0, 1]
        x = (x * 2) - 1
        return self.layers(x)


class RNDecoder(nn.Module):
    def __init__(self, in_nc=512, last_nc=128, tasks=["autoencoder"]):
        super().__init__()
        self.tasks = tasks
        out_dim = sum([task_to_out_nc[task] for task in tasks])

        nc = max(last_nc, min(in_nc, last_nc * 4))
        layers = [
            nn.ConvTranspose2d(in_nc, nc, [3, 3], 2, 1, bias=False, output_padding=1),
            nn.BatchNorm2d(nc),
            nn.ReLU(),
        ]
        for i in range(1, 3):
            layers += [
                nn.ConvTranspose2d(
                    nc,
                    max(last_nc, nc // 2),
                    [3, 3],
                    2,
                    1,
                    bias=False,
                    output_padding=1,
                ),
                nn.BatchNorm2d(max(last_nc, nc // 2)),
                nn.ReLU(),
            ]
            nc = max(last_nc, nc // 2)
        self.layers = nn.Sequential(*layers)

        self.heads = nn.ModuleDict()
        for task in tasks:
            self.heads[task] = nn.ConvTranspose2d(
                last_nc,
                task_to_out_nc[task],
                [3, 3],
                2,
                1,
                bias=False,
                output_padding=1,
            )

    def forward(self, x):
        out = self.layers(x)
        out = {task_name: out_activations[task_name](tasknet(out)) for task_name, tasknet in self.heads.items()}
        return out
