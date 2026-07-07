import os
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


class NAKO10KBodyDataset(Dataset):
    def __init__(self, patient_ids, body_data, mode,
                 query_samples_dir=None, metadata_csv =None

    ):
        if mode not in ("train","test"):
            raise ValueError(
                "mode must be 'train' or 'test'"
            )

        if mode == "train" and query_samples_dir is None:
            raise ValueError(
                "query_samples_dir must be provided"
            )

        if mode == "test" and metadata_csv is None:
            raise ValueError(
                "metadata_csv must be provided"
            )
        self.patient_ids = [str(pid) for pid in patient_ids]
        self.body_data = body_data
        self.mode = mode
        self.query_samples_dir = query_samples_dir

        self.metadata = None

        if metadata_csv is not None:
            self.metadata = pd.read_csv(
                metadata_csv,
                dtype={"patient_id": str},
            ).set_index("patient_id")

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index):
        patient_id = self.patient_ids[index]
        body_key = f"{patient_id}__input_points"

        if body_key not in self.body_data.files:
            raise ValueError(
                f"Patient {patient_id}: body points not found"
            )

        sample = {
            "body_points": torch.from_numpy(
                self.body_data[body_key].astype(np.float32)
            ),
            "patient_id": patient_id,
        }

        if self.mode == "train":
            query_path = os.path.join(
                self.query_samples_dir,
                f"{patient_id}.npz",
            )

            if not os.path.exists(query_path):
                raise ValueError(
                    f"Patient {patient_id}: query file not found"
                )

            with np.load(query_path, allow_pickle=False) as data:
                sample["query_points"] = torch.from_numpy(
                    data["query_points_dpt"].astype(np.float32)
                )
                sample["query_labels"] = torch.from_numpy(
                    data["query_train_labels"].astype(np.int64)
                )

        else:
            if patient_id not in self.metadata.index:
                raise ValueError(
                    f"Patient {patient_id}: metadata not found"
                )

            row = self.metadata.loc[patient_id]

            sample["dim"] = torch.tensor(
                [
                    int(row["shape_z"]),
                    int(row["shape_y"]),
                    int(row["shape_x"]),
                ],
                dtype=torch.int64,
            )
            sample["spacing"] = torch.tensor(
                [row["spacing_x"], row["spacing_y"], row["spacing_z"]],
                dtype=torch.float32,
            )
            sample["origin"] = torch.tensor(
                [row["origin_x"], row["origin_y"], row["origin_z"]],
                dtype=torch.float32,
            )
            sample["direction"] = torch.tensor(
                [row[f"direction_{i}"] for i in range(9)],
                dtype=torch.float32,
            )

        return sample


def collate_fn(batch):
    body_coord = [
        sample["body_points"] for sample in batch
    ]

    body_offsets = []
    count = 0

    for points in body_coord:
        count += points.shape[0]
        body_offsets.append(count)

    data = {
        "body_points": torch.cat(body_coord, dim=0).float(),
        "body_offsets": torch.tensor(
            body_offsets,
            dtype=torch.int32,
        ),
        "patient_ids": [
            sample["patient_id"] for sample in batch
        ],
    }

    if "query_points" in batch[0]:
        data["query_points"] = torch.stack([
            sample["query_points"] for sample in batch
        ]).float()

        data["query_labels"] = torch.stack([
            sample["query_labels"] for sample in batch
        ]).long()

    if "dim" in batch[0]:
        for key in ("dim", "spacing", "origin", "direction"):
            data[key] = torch.stack([
                sample[key] for sample in batch
            ])

    return data