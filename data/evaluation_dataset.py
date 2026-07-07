import os
import torch
import numpy as np
import SimpleITK as sitk

from torch.utils.data import Dataset


class NAKO10KBodyEvaluationDataset(Dataset):
    def __init__(self, patient_ids, body_data, mask_dir):
        self.patient_ids = list(patient_ids)
        self.body_data = body_data
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index):
        patient_id = self.patient_ids[index]

        body_points = self.body_data[
            f"{patient_id}__input_points"
        ].astype(np.float32)

        mask_path = os.path.join(self.mask_dir, f"{patient_id}.nii.gz")

        if not os.path.exists(mask_path):
            raise ValueError(
                f"Patient {patient_id}: mask file not found"
            )

        img = sitk.ReadImage(mask_path)
        seg = sitk.GetArrayFromImage(img).astype(np.int64)

        spacing = np.asarray(img.GetSpacing(), dtype=np.float32)
        origin = np.asarray(img.GetOrigin(), dtype=np.float32)
        direction = np.asarray(img.GetDirection(), dtype=np.float32)

        return (
            torch.from_numpy(body_points),
            torch.from_numpy(seg),
            patient_id,
            torch.from_numpy(spacing),
            torch.from_numpy(origin),
            torch.from_numpy(direction),

        )

def evaluation_collate_fn(batch):
    body_coord, segmentations, patient_ids, spacings, origins, directions = zip(*batch)

    body_offsets = []
    count = 0

    for points in body_coord:
        count += points.shape[0]
        body_offsets.append(count)

    return {
        "body_points": torch.cat(body_coord, dim=0).float(),
        "body_offsets": torch.tensor(body_offsets, dtype=torch.int32),
        "segmentations": torch.stack(segmentations),
        "patient_ids": list(patient_ids),
        "spacings": torch.stack(spacings),
        "origins": torch.stack(origins),
        "directions": torch.stack(directions),
    }