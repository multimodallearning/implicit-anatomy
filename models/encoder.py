import torch
import torch.nn as nn

from lib.pointops.functions import pointops

def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class BodyPointTransformerEncoder(nn.Module):
    def __init__(self, block, cfg_body, c=3):
        super().__init__()

        # Parameters
        self.num_encoder = cfg_body.num_encoder
        self.planes = cfg_body.planes
        self.blocks = cfg_body.blocks
        self.share_planes = cfg_body.share_planes
        self.stride = cfg_body.stride
        self.nsample = cfg_body.nsample

        self.c = c
        self.in_planes = c if c > 3 else 3
        self.encoders = []
        for i in range(self.num_encoder):
            enc = self._make_enc(block, self.planes[i], self.blocks[i], self.share_planes, stride=self.stride[i], nsample=self.nsample[i])
            self.encoders.append(enc)
        self.encoders = nn.ModuleList(self.encoders)

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_planes,
                    self.in_planes,
                    share_planes=share_planes,
                    nsample=nsample,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, pxo):
        p0, x0, o0 = pxo # points (n,3), features (n, c), offset (b)

        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)

        features_p, features_x, features_o = [p0], [x0], [o0]
        for i in range(self.num_encoder):
            p0, x0, o0 = self.encoders[i]([p0, x0, o0])
            features_p.append(p0)
            features_x.append(x0)
            features_o.append(o0)


        return {
            "points": features_p,
            "features": features_x,
            "offsets": features_o,
        }


class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True),
                                      nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                      nn.Linear(mid_planes, mid_planes // share_planes),
                                      nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                      nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)
        x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)  # (n, nsample, 3+c)
        x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)  # (n, nsample, c)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        for i, layer in enumerate(self.linear_p): p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1,
                                                                                                          2).contiguous() if i == 1 else layer(
            p_r)  # (n, nsample, c)
        w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes,
                                              self.mid_planes).sum(2)  # (n, nsample, c)
        for i, layer in enumerate(self.linear_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1,
                                                                                                      2).contiguous() if i % 3 == 0 else layer(
            w)
        w = self.softmax(w)  # (n, nsample, c)
        n, nsample, c = x_v.shape;
        s = self.share_planes
        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        return x

class PointTransformerLayerLN(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample

        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)

        self.linear_p = nn.Sequential(
            nn.Linear(3, 3),
            nn.LayerNorm(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_planes),
        )

        self.linear_w = nn.Sequential(
            nn.LayerNorm(mid_planes),
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes, mid_planes // share_planes),
            nn.LayerNorm(mid_planes // share_planes),
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes // share_planes, out_planes // share_planes),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo

        x_q = self.linear_q(x)
        x_k = self.linear_k(x)
        x_v = self.linear_v(x)

        x_k = pointops.queryandgroup(
            self.nsample, p, p, x_k, None, o, o, use_xyz=True
        )
        x_v = pointops.queryandgroup(
            self.nsample, p, p, x_v, None, o, o, use_xyz=False
        )

        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]

        p_r = self.linear_p(p_r)

        w = x_k - x_q.unsqueeze(1) + p_r.view(
            p_r.shape[0],
            p_r.shape[1],
            self.out_planes // self.mid_planes,
            self.mid_planes,
        ).sum(2)

        w = self.linear_w(w)
        w = self.softmax(w)

        n, nsample, c = x_v.shape
        s = self.share_planes
        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)

        return x

class PointTransformerBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(PointTransformerBlock, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]

class PointTransformerBlockLN(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(PointTransformerBlockLN, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.norm1 = nn.LayerNorm(planes)
        self.transformer2 = PointTransformerLayerLN(planes, planes, share_planes, nsample)
        self.norm2 = nn.LayerNorm(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.norm3 = nn.LayerNorm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.norm1(self.linear1(x)))
        x = self.relu(self.norm2(self.transformer2([p, x, o])))
        x = self.norm3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]

class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.furthestsampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)
            x = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)  # (m, 3+c, nsample)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))  # (m, c, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
        return [p, x, o]


class GlobalBodyEncoder(nn.Module):
    """
    Encodes a body-surface point cloud into one global latent vector
    per patient using a DPT encoder followed by max pooling.
    """

    def __init__(self, cfg_body, c=3):
        super().__init__()

        self.encoder = BodyPointTransformerEncoder(
            PointTransformerBlock,
            cfg_body,
            c=c,
        )

    @staticmethod
    def _global_max_pool(features, offsets):
        patient_codes = []
        start = 0

        for end in offsets:
            end = int(end.item())
            patient_features = features[start:end]

            if patient_features.shape[0] == 0:
                raise ValueError(
                    "Patient has no bottleneck features."
                )

            patient_code = maxpool(
                patient_features,
                dim=0,
            )

            patient_codes.append(patient_code)
            start = end

        return torch.stack(patient_codes, dim=0)

    def forward(self, body_points, body_offsets):
        encoded = self.encoder(
            [body_points, body_points, body_offsets]
        )

        bottleneck_features = encoded["features"][-1]
        bottleneck_offsets = encoded["offsets"][-1]

        patient_codes = self._global_max_pool(
            bottleneck_features,
            bottleneck_offsets,
        )

        return patient_codes

class GlobalBodyEncoderLN(nn.Module):
    """
    PointTransformer encoder using LayerNorm in attention blocks,
    BatchNorm in TransitionDown, and global max pooling.
    """

    def __init__(self, cfg_body, c=3):
        super().__init__()

        self.encoder = BodyPointTransformerEncoder(
            PointTransformerBlockLN,
            cfg_body,
            c=c,
        )

    @staticmethod
    def _global_max_pool(features, offsets):
        patient_codes = []
        start = 0

        for end in offsets:
            end = int(end.item())
            patient_features = features[start:end]

            if patient_features.shape[0] == 0:
                raise ValueError(
                    "Patient has no bottleneck features."
                )

            patient_code = maxpool(
                patient_features,
                dim=0,
            )

            patient_codes.append(patient_code)
            start = end

        return torch.stack(patient_codes, dim=0)

    def forward(self, body_points, body_offsets):
        encoded = self.encoder(
            [body_points, body_points, body_offsets]
        )

        bottleneck_features = encoded["features"][-1]
        bottleneck_offsets = encoded["offsets"][-1]

        patient_codes = self._global_max_pool(
            bottleneck_features,
            bottleneck_offsets,
        )

        return patient_codes
