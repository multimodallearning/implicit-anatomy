import os

import numpy as np
import pandas as pd
import SimpleITK as sitk

from tqdm import tqdm

from common.scoring import compute_hausdorff, compute_dice_per_organ, compute_detection_offset_error, compute_surface_distance


class EvaluateSegmentations:

    metrics = ['DSC', 'HDD', 'HDD95', 'DOE', 'USD', 'ASSD']

    metric_names = {
        'DSC'   : 'Dice coefficient',
        'HDD'   : 'Hausdorff distance',
        'HDD95' : '95th percentile Hausdorff distance',
        'DOE'   : 'Detection offset error',
        'USD'   : 'Unweighted surface distance',
        'ASSD'  : 'Average symmetric surface distance',

    }
    def __init__(
        self,
        gt_dir,
        pred_dir,
        patient_ids,
        name_mapping,
        selected_metrics,
        metrics_dir,
    ):
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir
        self.patient_ids = [str(pid).strip() for pid in patient_ids]
        self.selected_metrics = selected_metrics
        self.metrics_dir = metrics_dir

        self.classes = [
            name_mapping[label]
            for label in sorted(name_mapping)
        ]


    def eval_seg(self, gt_path, pred_path):
        gt = sitk.ReadImage(gt_path, sitk.sitkUInt8)
        pred = sitk.ReadImage(pred_path, sitk.sitkUInt8)

        geometry_matches = (
            pred.GetSize() == gt.GetSize()
            and np.allclose(pred.GetSpacing(), gt.GetSpacing())
            and np.allclose(pred.GetOrigin(), gt.GetOrigin())
            and np.allclose(pred.GetDirection(), gt.GetDirection())
        )

        if not geometry_matches:
            raise RuntimeError(
                "Geometry of ground-truth and prediction does not match."
            )

        results = {}

        if any(metric in self.selected_metrics for metric in ['HDD', 'HDD95']):
            results.update(
                compute_hausdorff(gt, pred, self.classes, percentile=95)
            )

        if "DSC" in self.selected_metrics:
            results.update(
                compute_dice_per_organ(gt, pred, self.classes)
            )

        if any(metric in self.selected_metrics for metric in ['USD', 'ASSD']):
            results.update(
                compute_surface_distance(gt, pred, self.classes)
            )

        if "DOE" in self.selected_metrics:
            results.update(
                compute_detection_offset_error(gt, pred, self.classes)
            )

        return results

    def eval_patient(self, patient_id):
        gt_path = os.path.join(
            self.gt_dir, f"{patient_id}.nii.gz"
        )
        pred_path = os.path.join(
            self.pred_dir, f"{patient_id}.nii.gz"
        )

        if not os.path.exists(gt_path):
            raise FileNotFoundError(gt_path)

        if not os.path.exists(pred_path):
            raise FileNotFoundError(pred_path)

        patient_metrics = {"patient_id": patient_id}
        patient_metrics.update(
            self.eval_seg(gt_path, pred_path)
        )

        return patient_metrics

    def eval_all(self):
        patient_data = []

        for patient_id in tqdm(
                self.patient_ids,
                desc="Evaluating",
                unit="patient",
        ):
            patient_data.append(
                self.eval_patient(patient_id)
            )

        df = pd.DataFrame(patient_data)
        metric_columns = [
            column
            for column in df.columns
            if column != "patient_id"
        ]

        os.makedirs(self.metrics_dir, exist_ok=True)

        df.to_csv(
            os.path.join(self.metrics_dir, "eval_all.csv"),
            index=False,
        )

        total_columns = ["patient_id"] + [
            column
            for column in metric_columns
            if "tot" in column
        ]

        df[total_columns].to_csv(
            os.path.join(self.metrics_dir, "eval.csv"),
            index=False,
        )

        return df






