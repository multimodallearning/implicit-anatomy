import numpy as np
import SimpleITK as sitk

from surface_distance import metrics

from util.tools import keep_one_label

def compute_surface_distance(image1, image2, classes):
    results = {}
    assd_values = []

    target = sitk.GetArrayViewFromImage(image1)
    prediction = sitk.GetArrayViewFromImage(image2)
    spacing_zyx = image1.GetSpacing()[::-1]

    for label, class_name in enumerate(classes[1:], start=1):
        target_mask = target == label
        prediction_mask = prediction == label

        if not target_mask.any() or not prediction_mask.any():
            results[f"assd_{class_name}"] = np.nan
            continue

        distances = metrics.compute_surface_distances(
            target_mask,
            prediction_mask,
            spacing_zyx,
        )

        weighted_gt_to_pred, weighted_pred_to_gt = (
            metrics.compute_average_surface_distance(distances)
        )
        assd = 0.5 * (
            weighted_gt_to_pred + weighted_pred_to_gt
        )

        results[f"assd_{class_name}"] = float(assd)

        assd_values.append(assd)

    results["assd_tot"] = (
        float(np.mean(assd_values))
        if assd_values else np.nan
    )

    return results

def compute_hausdorff(image1, image2, classes, percentile=95.0):
    results = {}
    hd_values = []
    hd95_values = []

    target = sitk.GetArrayViewFromImage(image1)
    prediction = sitk.GetArrayViewFromImage(image2)
    spacing_zyx = image1.GetSpacing()[::-1]

    for label, class_name in enumerate(classes[1:], start=1):
        target_mask = target == label
        prediction_mask = prediction == label

        if not np.any(target_mask) or not np.any(prediction_mask):
            results[f"hd_{class_name}"] = np.nan
            results[f"hd95_{class_name}"] = np.nan
            continue

        target_label = keep_one_label(
            image1, label, len(classes)
        )
        prediction_label = keep_one_label(
            image2, label, len(classes)
        )

        hausdorff_filter = sitk.HausdorffDistanceImageFilter()
        hausdorff_filter.Execute(target_label, prediction_label)
        hdd_value = hausdorff_filter.GetHausdorffDistance()

        surface_distances = metrics.compute_surface_distances(
            target_mask,
            prediction_mask,
            spacing_zyx,
        )

        distances_gt_to_pred = surface_distances['distances_gt_to_pred']
        distances_pred_to_gt = surface_distances['distances_pred_to_gt']

        # Calculate 95th percentile of the Hausdorff distance
        hd95_gt_to_pred = np.percentile(distances_gt_to_pred, percentile)
        hd95_pred_to_gt = np.percentile(distances_pred_to_gt, percentile)
        hd95_value = max(hd95_gt_to_pred, hd95_pred_to_gt)

        results[f"hd_{class_name}"] = float(hdd_value)
        results[f"hd95_{class_name}"] = float(hd95_value)

        hd_values.append(hdd_value)
        hd95_values.append(hd95_value)

    results["hd_tot"] = (
        float(np.mean(hd_values)) if hd_values else np.nan
    )
    results["hd95_tot"] = (
        float(np.mean(hd95_values)) if hd95_values else np.nan
    )

    return results

def compute_dice_per_organ(image1, image2, classes):
    target = sitk.GetArrayFromImage(image1)
    inputs = sitk.GetArrayFromImage(image2)

    dice = {}
    dice_values = []

    for label, class_name in enumerate(classes[1:], start=1):
        target_mask = target == label
        input_mask = inputs == label

        target_sum = target_mask.sum()
        input_sum = input_mask.sum()

        if target_sum == 0 and input_sum == 0:
            dice_value = np.nan
        else:
            intersection = np.logical_and(
                target_mask, input_mask
            ).sum()

            dice_value = (
                2.0 * intersection
                / (target_sum + input_sum)
            )
            dice_values.append(dice_value)

        dice[f"dice_{class_name}"] = dice_value

    dice["dice_tot"] = (
        float(np.mean(dice_values))
        if dice_values
        else np.nan
    )

    return dice
