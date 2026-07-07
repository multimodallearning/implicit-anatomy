from typing import Dict

import os
import json
import numpy as np
import SimpleITK as sitk

from scipy.ndimage import binary_erosion

from common.dpt_coordinates import voxel_zyx_to_dpt


def load_train_label_json(path) -> Dict[int, str]:
    with open(path, "r") as f:
        train_id_to_organ = json.load(f)

    return {
        int(train_id): organ
        for train_id, organ in train_id_to_organ.items()
    }

def get_patient_ids(path):
    mask_files = sorted(
        f for f in os.listdir(path)
        if f.endswith(".nii.gz")
    )

    return [
        f.replace(".nii.gz", "")
        for f in mask_files
    ]

class QuerySampling:
    """
    Samples DPT-normalized query points from volumetric organ segmentation masks.

    Internally, train label 0 is reserved for background. Therefore organs from
    the train JSON are shifted by +1:
        background -> 0
        kidney_right -> 1
        kidney_left -> 2
        liver -> 3

    :param mask_dir: Directory containing mapped masks named '<patient_id>.nii.gz'.
    :param train_label_json_path: JSON mapping train order ids to organ names.
    :param n_per_organ: Base number of samples per organ. 90% are surface base
        points, each producing one sample per sigma. 10% are bbox samples.
    :param near_surface_ratio: Fraction used for surface base points.
    :param n_global_background: Number of background voxels sampled
    uniformly from the complete volume, once per patient.
    :param surface_sigmas: Gaussian jitter sigmas in voxel units.
    :param normalize_to_dpt: If True, additionally stores query points in the
    normalized DPT coordinate system.
    :param log_ratio: If True, prints inside-ratio per organ.
    """

    # TODO: subsample query points for the computation of metric

    def __init__(
        self,
        mask_dir: str,
        train_label_json_path: str,
        n_per_organ: int = 1024,
        near_surface_ratio: float = 0.9,
        n_global_background: int = 4096,
        surface_sigmas=(1.0, 3.0),
        normalize_to_dpt=False,
        log_ratio: bool = False,
    ):
        self.mask_dir = mask_dir
        self.train_label_json_path = train_label_json_path
        self.n_per_organ = n_per_organ
        self.near_surface_ratio = near_surface_ratio
        self.n_global_background = n_global_background
        self.surface_sigmas = surface_sigmas
        self.normalize_to_dpt=normalize_to_dpt
        self.log_ratio = log_ratio

        self.train_id_to_organ = load_train_label_json(self.train_label_json_path)

        self.train_label_to_organ = {0: "background"}

        for train_id, organ in self.train_id_to_organ.items():
            train_label = train_id + 1
            self.train_label_to_organ[train_label] = organ

    def _sample_global_background(self, seg, n_samples):
        flat_seg = seg.reshape(-1)
        background_indices = np.flatnonzero(flat_seg == 0)

        if background_indices.size == 0:
            raise ValueError("Segmentation contains no background voxels.")

        replace = background_indices.size < n_samples

        selected_flat_indices = np.random.choice(
            background_indices,
            size=n_samples,
            replace=replace,
        )

        samples = np.column_stack(
            np.unravel_index(
                selected_flat_indices,
                seg.shape,
            )
        ).astype(np.float32)

        labels = np.zeros(
            n_samples,
            dtype=np.uint8,
        )

        return samples, labels

    def _sample_one_organ(self, seg, organ, train_label):
        n_surface_base = int(round(self.n_per_organ * self.near_surface_ratio))
        n_bbox = self.n_per_organ - n_surface_base

        organ_mask = seg == train_label
        organ_voxels = np.argwhere(organ_mask)

        if organ_voxels.shape[0] == 0:
            raise ValueError(f"{organ} / train label {train_label}: not found")

        eroded = binary_erosion(organ_mask, iterations=1)
        surface_mask = organ_mask & (~eroded)
        surface_voxels = np.argwhere(surface_mask)

        if surface_voxels.shape[0] == 0:
            raise ValueError(f"{organ} / train label {train_label}: no surface voxels")

        surface_idx = np.random.randint(
            0,
            surface_voxels.shape[0],
            size=n_surface_base,
        )
        surface_base = surface_voxels[surface_idx].astype(np.float32)

        surface_parts = []

        for sigma in self.surface_sigmas:
            jitter = np.random.normal(
                loc=0.0,
                scale=sigma,
                size=surface_base.shape,
            ).astype(np.float32)

            surface_parts.append(surface_base + jitter)

        surface_samples = np.concatenate(surface_parts, axis=0)

        max_zyx_float = np.array(seg.shape, dtype=np.float32) - 1
        surface_samples = np.clip(surface_samples, 0, max_zyx_float)

        bbox_min = organ_voxels.min(axis=0)
        bbox_max = organ_voxels.max(axis=0) + 1

        bbox_samples = np.stack(
            [
                np.random.randint(bbox_min[d], bbox_max[d], size=n_bbox)
                for d in range(3)
            ],
            axis=1,
        ).astype(np.float32)

        samples = np.concatenate([surface_samples, bbox_samples], axis=0)

        sample_indices = np.rint(samples).astype(np.int64)
        max_zyx_int = np.array(seg.shape, dtype=np.int64) - 1
        sample_indices = np.clip(sample_indices, 0, max_zyx_int)

        sampled_labels = seg[
            sample_indices[:, 0],
            sample_indices[:, 1],
            sample_indices[:, 2],
        ]

        if self.log_ratio:
            inside_ratio = np.mean(sampled_labels == train_label)
            print(
                f"{organ:18s} "
                f"label={train_label:2d} "
                f"inside_ratio={inside_ratio:.3f}"
            )

        return samples, sampled_labels

    def _sample_segmentation(self, seg):
        query_voxels = []
        query_train_labels = []

        bg_samples, bg_labels = (self._sample_global_background(seg, self.n_global_background))
        query_voxels.append(bg_samples)
        query_train_labels.append(bg_labels)

        for train_id in sorted(self.train_id_to_organ.keys()):
            organ = self.train_id_to_organ[train_id]
            train_label = train_id + 1

            samples, labels = self._sample_one_organ(seg, organ, train_label)

            query_voxels.append(samples)
            query_train_labels.append(labels)

        if len(query_voxels) == 0:
            raise ValueError("No query samples were created.")

        query_voxels = np.concatenate(query_voxels, axis=0)
        query_train_labels = np.concatenate(query_train_labels, axis=0)

        return query_voxels, query_train_labels

    def sample_patient(self, patient_id):
        mask_path = os.path.join(
            self.mask_dir,
            f"{patient_id}.nii.gz",
        )

        if not os.path.exists(mask_path):
            raise ValueError(f"Mask not found: {mask_path}")

        img = sitk.ReadImage(mask_path)
        seg = sitk.GetArrayFromImage(img).astype(np.int16)


        query_voxels, query_train_labels = self._sample_segmentation(seg)

        result = {
            "query_voxels_zyx": query_voxels.astype(np.float32),
            "query_train_labels": query_train_labels.astype(np.uint8),
        }

        if self.normalize_to_dpt:
            result["query_points_dpt"] = voxel_zyx_to_dpt(
                query_voxels,
                seg.shape,
            )

        return result

    def sample(self, output_dir, patient_ids):
        os.makedirs(output_dir, exist_ok=True)

        for patient_id in patient_ids:
            save_path = os.path.join(
                output_dir,
                f"{patient_id}.npz",
            )

            if os.path.exists(save_path):
                continue

            try:
                samples = self.sample_patient(patient_id)
            except ValueError as error:
                raise ValueError(
                    f"Patient {patient_id}: {error}"
                ) from error

            np.savez_compressed(save_path, **samples)












