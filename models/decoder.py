import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sconet.layers import ResnetBlockFC
from siren_pytorch import SirenNet


class GlobalDecoder(nn.Module):
    """
    Classifies DPT-normalized query points using one global body code
    per patient.

    query_points: (B, N, 3)
    body_codes:   (B, 512)
    output:       (B, N, num_classes)
    """

    def __init__(
        self,
        dim=3,
        c_dim=512,
        hidden_size=256,
        n_blocks=5,
        num_classes=21,
        leaky=False,
    ):
        super().__init__()

        self.c_dim = c_dim
        self.n_blocks = n_blocks

        self.fc_p = nn.Linear(dim, hidden_size)

        self.fc_c = nn.ModuleList([
            nn.Linear(c_dim, hidden_size)
            for _ in range(n_blocks)
        ])

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size)
            for _ in range(n_blocks)
        ])

        self.fc_out = nn.Linear(
            hidden_size,
            num_classes,
        )

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, query_points, body_codes):
        net = self.fc_p(query_points)

        for i in range(self.n_blocks):
            net_c = self.fc_c[i](body_codes).unsqueeze(1)
            net = net + net_c
            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        return out

# modulatory feed forward

class Modulator(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.ReLU()
            ))

    def forward(self, z):
        # z: [B, dim_in]
        x = z
        hiddens = []

        for layer in self.layers:
            x = layer(x)  # [B, dim_hidden]
            hiddens.append(x)
            x = torch.cat((x, z), dim=-1)

        return tuple(hiddens)


class ModulatedSirenDecoder(nn.Module):
    def __init__(
        self,
        dim=3,
        c_dim=512,
        hidden_size=256,
        n_layers=5,
        num_classes=21,
        w0_initial=30.,
    ):
        super().__init__()

        self.net = SirenNet(
            dim_in=dim,
            dim_hidden=hidden_size,
            dim_out=num_classes,
            num_layers=n_layers,
            w0_initial=w0_initial,
            final_activation=nn.Identity(),
        )

        self.modulator = Modulator(
            dim_in=c_dim,
            dim_hidden=hidden_size,
            num_layers=n_layers,
        )

    def forward(self, query_points, body_codes):
        # query_points: [B, N, 3]
        # body_codes:   [B, 512]

        mods = self.modulator(body_codes)
        x = query_points

        for layer, modulation in zip(self.net.layers, mods):
            x = layer(x)  # [B, N, 256]
            x = x * modulation.unsqueeze(1)  # [B, 1, 256]

        return self.net.last_layer(x)  # [B, N, 21]

if __name__ == "__main__":
    device = torch.device("cuda:0")

    decoder = ModulatedSirenDecoder(
        dim=3,
        c_dim=512,
        hidden_size=256,
        n_layers=5,
        num_classes=21,
        w0_initial=30.,
    ).to(device)

    query_points = torch.randn(2, 1000, 3, device=device)
    body_codes = torch.randn(2, 512, device=device)

    logits = decoder(query_points, body_codes)

    print(logits.shape)
    print(torch.isfinite(logits).all().item())