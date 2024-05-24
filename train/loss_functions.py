import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class BCE_Dice_Loss(nn.Module):
    def __init__(self):
        super(BCE_Dice_Loss, self).__init__()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=False)

    def forward(self, inputs, targets, weights):
        dice_loss = self.dice(inputs, targets)
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, weight=weights)
        return dice_loss + bce_loss
