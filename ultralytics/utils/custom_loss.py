import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        targets = targets.float()
        # preds = torch.sigmoid(preds)

        intersection = (preds*targets).sum(dim=(1,2))
        union = targets.sum(dim=(1,2)) + preds.sum(dim=(1,2))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()

DcLoss = DiceLoss()