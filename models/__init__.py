import torch.nn as nn


class BodyImplicitSegmentationNetwork(nn.Module):
    """
    Predicts organ classes at query locations conditioned on a
    body-surface point cloud.
    """

    def __init__(self, decoder, encoder, device=None):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self._device = device

        if device is not None:
            self.to(device)

    def forward(
        self,
        query_points,
        body_points,
        body_offsets,
        **kwargs,
    ):
        body_codes = self.encode_inputs(
            body_points,
            body_offsets,
        )

        return self.decode(
            query_points,
            body_codes,
            **kwargs,
        )

    def encode_inputs(self, body_points, body_offsets):
        return self.encoder(
            body_points,
            body_offsets,
        )

    def decode(self, query_points, body_codes, **kwargs):
        return self.decoder(
            query_points,
            body_codes,
            **kwargs,
        )

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model