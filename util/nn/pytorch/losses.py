import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_ce_class_weights(query_samples_dir, patient_ids, num_classes):
    import torch
    import numpy as np
    import os

    counts = np.zeros(num_classes, dtype=np.int64)

    for patient_id in patient_ids:
        query_path = os.path.join(query_samples_dir, f"{patient_id}.npz")

        with np.load(query_path, allow_pickle=False) as data:
            labels = data["query_train_labels"].reshape(-1)

        counts += np.bincount(labels, minlength=num_classes)

    freq = counts / counts.sum()
    weights = 1.0 / np.sqrt(freq + 1e-12)
    weights = weights / weights.mean()

    #return torch.tensor(weights, dtype=torch.float32), counts, freq
    return torch.tensor(weights, dtype=torch.float32)


class CELoss(nn.Module):
    def __init__(self, w = 1, class_weights=None):
        super().__init__()
        self.w = w
        self.class_weights = class_weights

    def forward(self, output, target, deep_supervision=False):
        if not deep_supervision:
            output = torch.swapaxes(output,1,2)

        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(output.device)
        loss = F.cross_entropy(
                output, target, weight=weight, reduction='none')
        return self.w * loss


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5, w=1, do_bg=True):
        super().__init__()
        self.epsilon = epsilon
        self.w = w
        self.do_bg = do_bg

    def forward(self, output, target):
        output = F.softmax(output, dim=2)

        target_one_hot = F.one_hot(
            target.long(),
            num_classes=output.shape[2],
        ).float()

        intersection = output * target_one_hot
        intersection = intersection.sum(dim=(0,1))

        denominator = output.sum(dim=(0,1)) + target_one_hot.sum(dim=(0,1))

        dice = (
            2.0 * intersection + self.epsilon
        ) / (
            denominator + self.epsilon
        )

        if not self.do_bg:
            dice = dice[1:]

        return self.w * (1.0 - dice)

class CEDiceLoss(nn.Module):
    def __init__(self, alpha, beta, do_bg=True, class_weights=None):
        super().__init__()
        self.ce = CELoss(w=alpha, class_weights=class_weights)
        self.dice = DiceLoss(w=beta, do_bg=do_bg)

    def forward(self, output, target):
        ce = self.ce(output, target).mean()
        dice = self.dice(output, target).mean()
        #print(f"\tCE loss : {ce.item():.4f}, Dice loss : {dice.item():.4f}")
        return ce + dice

def _create_loss(name, params):
    if name == 'DiceLoss':
        do_bg = params['do_bg']
        return DiceLoss(do_bg=do_bg)
    elif name == 'CELoss':
        alpha = params['alpha']
        return CELoss(w=alpha)
    elif name == 'CEDiceLoss':
        alpha = params['alpha']
        beta = params['beta']
        do_bg = params['do_bg']
        return CEDiceLoss(alpha, beta, do_bg=do_bg)
    else:
        raise RuntimeError(f"Unsupported loss function: '{name}'")





