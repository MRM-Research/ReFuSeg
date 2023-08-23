import torch
from torch.nn.modules.loss import _Loss

class IoU(_Loss):
    def __init__(self, num_classes):
        super(IoU, self).__init__()
        self.num_classes = num_classes
        self.register_buffer('intersection', torch.zeros(num_classes))
        self.register_buffer('union', torch.zeros(num_classes))
    
    def forward(self, preds, targets):
        preds = torch.argmax(preds, dim=1)
        self.intersection = self.intersection.to(targets.device)
        self.union = self.union.to(targets.device)

        for cls in range(self.num_classes):
            output_mask = preds == cls
            target_mask = targets == cls

            class_intersection = (output_mask & target_mask).sum()
            class_union = (output_mask | target_mask).sum()

            self.intersection[cls] += class_intersection.float()
            self.union[cls] += class_union.float()

        iou = self.intersection / (self.union + 1e-7)
        mean_iou = iou.mean()

        return mean_iou