import numpy as np
import SimpleITK as sitk

def to_nii(volume, out_path, spacing, origin, direction):
    image = sitk.GetImageFromArray(
        volume.astype(np.int16)
    )
    image.SetSpacing(tuple(spacing))
    image.SetOrigin(tuple(origin))
    image.SetDirection(tuple(direction))
    sitk.WriteImage(image, out_path)


def keep_one_label(image, l, num_labels):
    '''
    Sets all labels except one to be background labels.
    Args:
        image (object): sitk image to process
        l (int): label to keep
        num_labels (int): original number of labels
    '''
    other_labels = [j for j in range(1, num_labels) if j != l]
    mapping = {}
    for j in other_labels:
        mapping[j] = 0

    image_l = change_label_nii(image, mapping)

    return image_l

def change_label_nii(img_nii,mapping):
    change_label_filter = sitk.ChangeLabelImageFilter()
    change_label_filter.SetChangeMap(mapping)
    img_new = change_label_filter.Execute(img_nii)
    return img_new