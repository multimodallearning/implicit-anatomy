import numpy as np
import torch


def voxel_zyx_to_dpt(voxels_zyx, shape_zyx):
    size_z, size_y, size_x = shape_zyx

    z = voxels_zyx[:, 0].astype(np.float32)
    y = voxels_zyx[:, 1].astype(np.float32)
    x = voxels_zyx[:, 2].astype(np.float32)

    return np.stack([
        2.0 * z / (size_z - 1) - 1.0,
        1.0 - 2.0 * y / (size_y - 1),
        1.0 - 2.0 * x / (size_x - 1),
    ], axis=1).astype(np.float32)


def make_dpt_grid(shape_zyx):
    """
    Create a 3D query grid.

    Args:
        shape_zyx (tuple): Volume shape given as (size_z, size_y, size_x).

    """
    size_z, size_y, size_x = shape_zyx
    size = size_z * size_y * size_x

    z_dpt = torch.linspace(-1.0, 1.0, size_z)
    y_dpt = torch.linspace(1.0, -1.0, size_y)
    x_dpt = torch.linspace(1.0, -1.0, size_x)

    z_dpt = z_dpt.view(-1, 1, 1).expand(*shape_zyx)
    y_dpt = y_dpt.view(1, -1, 1).expand(*shape_zyx)
    x_dpt = x_dpt.view(1, 1, -1).expand(*shape_zyx)

    return torch.stack(
        [z_dpt, y_dpt, x_dpt],
        dim=-1,
    ).reshape(size, 3)