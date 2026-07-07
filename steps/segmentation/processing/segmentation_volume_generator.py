import torch
import time

from fvcore.nn import FlopCountAnalysis

from common.dpt_coordinates import make_dpt_grid
from util.jit_handles import _SUPPORTED_OPS


class SegmentationVolumeGenerator(object):
    def __init__(self,  model, num_classes ,points_batch_size=100000,
                 device=None, compute_flops=False):
        self.model = model
        self.num_classes = num_classes
        self.points_batch_size = points_batch_size
        self.device = device
        self.compute_flops = compute_flops

    def generate_volume(self, data, dim):
        self.model.eval()

        device = self.device
        stats_dict = {}
        kwargs = {}

        body_points = data['body_points'].to(device)
        body_offsets = data['body_offsets'].to(device)

        t0 = time.time()
        with torch.inference_mode():
            body_codes = self.model.encode_inputs(
                body_points,
                body_offsets,
            )


            # print("DEBUG body_codes")
            # print("body_codes shape:", body_codes.shape)
            # print("body_codes min:", body_codes.min().item())
            # print("body_codes max:", body_codes.max().item())
            # print("body_codes mean:", body_codes.mean().item())
            # print("body_codes std:", body_codes.std().item())
            # print("body_codes nan:", torch.isnan(body_codes).any().item())
            # print("DEBUG body_offsets")
            # print("body_points shape:", body_points.shape)
            # print("body_offsets:", body_offsets)

            if self.compute_flops:
                flops = FlopCountAnalysis(self.model.encoder, (body_points, body_offsets,)).uncalled_modules_warnings(False).unsupported_ops_warnings(False).set_op_handle(**_SUPPORTED_OPS)
                self.flop_counter = flops.total()

        stats_dict['time (encode inputs)'] = time.time() - t0

        volume = self.generate_volume_from_latent(body_codes=body_codes,
                                                  dim=dim,
                                                  **kwargs)

        out = {}
        out['volume'] = volume
        out['stats_dict'] = stats_dict
        if self.compute_flops:
            out['flop_counter'] = self.flop_counter

        return out


    def generate_volume_from_latent(self, dim, body_codes=None, **kwargs):
        query_points = make_dpt_grid(dim)

        labels = self.eval_points(query_points, body_codes, **kwargs).numpy()

        return labels.reshape(dim)


    def eval_points(self, query_points, body_codes=None, **kwargs):
        ''' Evaluates the occupancy values for the points.

        Args:
            query_points (tensor): points
            body_codes (tensor): encoded feature volumes
        '''
        p_split = torch.split(query_points, self.points_batch_size)
        label_chunks = []
        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.inference_mode():
                logits= self.model.decode(pi, body_codes, **kwargs)



            labels = logits.argmax(dim=-1)
            label_chunks.append(labels.squeeze(0).cpu())

            if self.compute_flops :
                flops = FlopCountAnalysis(self.model.decoder, (pi,body_codes)).uncalled_modules_warnings(False).set_op_handle(**_SUPPORTED_OPS)
                self.flop_counter += flops.total()

        return torch.cat(label_chunks, dim=0)






