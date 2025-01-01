import torch
import torch.nn as nn

class OptimizedSSE(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        super(OptimizedSSE, self).__init__()
        self.S, self.B, self.C = S, B, C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, targets):
        # Separate components of predictions and targets
        pred_boxes = predictions[..., :self.B * 5].reshape(-1, self.S, self.S, self.B, 5)  # (x, y, w, h, conf)
        pred_classes = predictions[..., self.B * 5:]  # Class probabilities

        target_boxes = targets[..., :self.B * 5].reshape(-1, self.S, self.S, self.B, 5)
        target_classes = targets[..., self.B * 5:]  # Ground truth classes

        # Mask for object presence (confidence = 1 for objects)
        obj_mask = target_boxes[..., 4] > 0  # Mask for object presence
        noobj_mask = target_boxes[..., 4] == 0  # Mask for no-object cells

        # Localization loss: (x, y, sqrt(w), sqrt(h))
        box_loss = torch.sum(
            obj_mask * (
                (pred_boxes[..., 0] - target_boxes[..., 0]) ** 2 +
                (pred_boxes[..., 1] - target_boxes[..., 1]) ** 2 +
                (torch.sqrt(pred_boxes[..., 2]) - torch.sqrt(target_boxes[..., 2])) ** 2 +
                (torch.sqrt(pred_boxes[..., 3]) - torch.sqrt(target_boxes[..., 3])) ** 2
            )
        )

        # Confidence loss: Separate for object and no-object cells
        conf_loss_obj = torch.sum(
            obj_mask * (pred_boxes[..., 4] - target_boxes[..., 4]) ** 2
        )
        conf_loss_noobj = torch.sum(
            noobj_mask * (pred_boxes[..., 4]) ** 2
        )

        # Classification loss: Sum of squared errors for class probabilities
        class_loss = torch.sum(
            obj_mask[..., 0].unsqueeze(-1) * (pred_classes - target_classes) ** 2
        )

        # Total loss
        loss = (
            self.lambda_coord * box_loss +
            conf_loss_obj +
            self.lambda_noobj * conf_loss_noobj +
            class_loss
        )

        return loss

