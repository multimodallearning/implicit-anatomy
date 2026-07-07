import os

import numpy as np
import SimpleITK as sitk


class MapLabels:
    """
    Maps label values in volumetric segmentation masks to training labels.

    Each input mask is read from a NIfTI file, transformed according to
    ``value_mapping``, and written to ``output_dir``. Labels not included in
    ``value_mapping`` are replaced by ``default_value``. Spatial metadata,
    including spacing, origin, and direction, is preserved from the original
    image.

    Parameters
    ----------
    input_dir : str
        Directory containing input masks named ``<patient_id>.nii.gz``.
    output_dir : str
        Directory in which the mapped masks are saved.
    value_mapping : dict
        Mapping from original label values to training label values.
    name_mapping : dict
        Mapping from training label values to human-readable class names.
        This mapping is retained for documentation and logging purposes.
    default_value : int, optional
        Value assigned to labels that do not occur in ``value_mapping``.
        The default is 0, which usually represents the background class.
    """

    def __init__(
        self,
        input_dir,
        output_dir,
        value_mapping,
        name_mapping,
        default_value=0

    ):
        self.input_dir = input_dir
        self.output_dir = output_dir

        self.value_mapping = {
            int(raw_label): int(train_label)
            for raw_label, train_label in value_mapping.items()
        }
        self.name_mapping = {
            int(train_label): organ
            for train_label, organ in name_mapping.items()
        }

        self.default_value = int(default_value)

    def _map_array(self, raw_seg):
        mapped_seg = np.full(
            raw_seg.shape,
            self.default_value,
            dtype=np.uint8,
        )

        for raw_label, train_label in self.value_mapping.items():
            mapped_seg[raw_seg == raw_label] = train_label

        return mapped_seg

    def map_patient(self, patient_id):
        input_path = os.path.join(
            self.input_dir,
            f"{patient_id}.nii.gz",
        )
        output_path = os.path.join(
            self.output_dir,
            f"{patient_id}.nii.gz",
        )

        if not os.path.exists(input_path):
            raise ValueError(f"Mask not found: {input_path}")

        raw_image = sitk.ReadImage(input_path)
        raw_seg = sitk.GetArrayFromImage(raw_image).astype(np.int64)

        mapped_seg = self._map_array(raw_seg)

        mapped_image = sitk.GetImageFromArray(mapped_seg)
        mapped_image.CopyInformation(raw_image)

        sitk.WriteImage(mapped_image, output_path)

    def process(self, patient_ids):
        os.makedirs(self.output_dir, exist_ok=True)

        for patient_id in patient_ids:
            output_path = os.path.join(
                self.output_dir,
                f"{patient_id}.nii.gz",
            )

            if os.path.exists(output_path):
                continue

            try:
                self.map_patient(patient_id)
            except ValueError as error:
                raise ValueError(
                    f"Patient {patient_id}: {error}"
                ) from error

