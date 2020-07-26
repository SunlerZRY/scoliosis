import torch
import torch.nn as nn
import skfuzzy as fuzz


# @torch.no_grad()
# def smooth_one_hot(true_labels: torch.Tensor, cutoff=13, width=1.6):
#     """
#     Only for two class
#     cutoff is the class cutoff point
#     Warning: This function has no grad.
#     """
#     # assert 0 <= smoothing < 1
#     somothing_confidence = fuzz.sigmf(true_labels.cpu().numpy(), cutoff, width)
#     somothing_confidence = torch.from_numpy(somothing_confidence)
#     label_shape = torch.Size((true_labels.size(0), 2))

#     smooth_label = torch.empty(size=label_shape, device=true_labels.device)
#     smooth_label[:, 0] = 1.0 - somothing_confidence
#     smooth_label[:, 1] = somothing_confidence
#     return smooth_label


# class LabelSmoothingLoss(nn.Module):
#     """This is label smoothing loss function.
#     """

#     def __init__(self, cutoff=13, width=1.6, dim=-1):
#         super(LabelSmoothingLoss, self).__init__()
#         self.width = width
#         self.cutoff = cutoff
#         self.dim = dim

#     def forward(self, pred, target):
#         pred = pred.log_softmax(dim=self.dim)
#         true_dist = smooth_one_hot(target, self.cutoff, self.width)
#         return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
@torch.no_grad()
def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    Warning: This function has no grad.
    """
    # assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))

    smooth_label = torch.empty(size=label_shape, device=true_labels.device)
    smooth_label.fill_(smoothing / (classes - 1))
    smooth_label.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return smooth_label


class LabelSmoothingLoss(nn.Module):
    """This is label smoothing loss function.
    """

    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        true_dist = smooth_one_hot(target, self.cls, self.smoothing)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
