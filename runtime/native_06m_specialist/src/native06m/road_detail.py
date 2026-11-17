"""Non-deployable road-detail experiment with a frozen five-class backbone."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


STAGE = "limited_geography_road_detail_development_pilot"


def conv_block(inputs, outputs, dilation=1):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=dilation, dilation=dilation, bias=False),
        nn.GroupNorm(4, outputs),
        nn.SiLU(),
    )


def road_margin(logits):
    """Binary road-versus-all-other-classes log odds, without changing priors."""
    return logits[:, 1:2] - torch.logsumexp(logits[:, [0, 2, 3, 4]], dim=1, keepdim=True)


class RoadDetailHead(nn.Module):
    def __init__(self, in_channels=4, residual_bound=4.0):
        super().__init__()
        if residual_bound <= 0:
            raise ValueError("Residual bound must be positive")
        self.residual_bound = residual_bound
        self.detail = nn.Sequential(conv_block(in_channels, 16), conv_block(16, 16, 2))
        self.stem_projection = nn.Sequential(nn.Conv2d(64, 8, 1), nn.SiLU())
        self.decoder_projection = nn.Sequential(nn.Conv2d(16, 8, 1), nn.SiLU())
        self.fusion = nn.Sequential(conv_block(37, 32), conv_block(32, 16, 3),
                                    conv_block(16, 16, 5))
        self.output = nn.Conv2d(16, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, image, stem, decoded, logits):
        stem = F.interpolate(self.stem_projection(stem), size=image.shape[-2:],
                             mode="bilinear", align_corners=False)
        context = self.decoder_projection(decoded)
        features = torch.cat((self.detail(image), stem, context, logits.softmax(1)), dim=1)
        raw = self.output(self.fusion(features))
        delta = self.residual_bound * torch.tanh(raw / self.residual_bound)
        # Only the road logit may move; non-road class ordering stays identical.
        correction = torch.cat((torch.zeros_like(delta), delta,
                                torch.zeros_like(delta).expand(-1, 3, -1, -1)), dim=1)
        return {"semantic": logits + correction, "road_delta": delta,
                "road": road_margin(logits) + delta,
                "baseline_road_probability": logits.softmax(1)[:, 1:2].detach(),
                "baseline_nonroad_confidence": logits.softmax(1)[:, [0, 2, 3, 4]].amax(1, keepdim=True).detach()}


class FrozenRoadDetailModel(nn.Module):
    def __init__(self, backbone, residual_bound=4.0):
        super().__init__()
        self.backbone = backbone
        self.head = RoadDetailHead(backbone.in_channels, residual_bound)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, image):
        with torch.no_grad():
            self.backbone.network.check_input_shape(image)
            features = self.backbone.network.encoder(image)
            decoded = self.backbone.network.decoder(features)
            logits = self.backbone.network.segmentation_head(decoded)
        return self.head(image, features[1], decoded, logits)


def road_detail_loss(output, target, positive_weight=2.0, dice_weight=.3,
                     residual_penalty=.005, focal_gamma=0.0):
    valid = target != 255
    if not valid.any():
        return output["road"].sum() * 0
    truth = (target == 1).float()
    margin = output["road"][:, 0]
    bce_pixels = F.binary_cross_entropy_with_logits(
        margin[valid], truth[valid], reduction="none",
        pos_weight=margin.new_tensor(positive_weight))
    if focal_gamma < 0:
        raise ValueError("Focal gamma cannot be negative")
    if focal_gamma:
        probability_true = torch.where(truth[valid].bool(), margin[valid].sigmoid(),
                                       (-margin[valid]).sigmoid())
        bce_pixels = bce_pixels * (1 - probability_true).pow(focal_gamma)
    bce = bce_pixels.mean()
    # An unreviewed road is not a negative; Dice sees reviewed pixels only.
    probability = margin.sigmoid()[valid]
    reference = truth[valid]
    dice = 1 - (2 * (probability * reference).sum() + 1) / (
        probability.sum() + reference.sum() + 1) if reference.any() else margin.sum() * 0
    regularization = output["road_delta"][:, 0][valid].square().mean()
    return bce + dice_weight * dice + residual_penalty * regularization


def candidate_guard(baseline, candidate, *, road_gain=.01, nonroad_tolerance=.01):
    """Fixed development acceptance rule, not evidence of cross-city accuracy."""
    failures = []
    if candidate["class_iou"][1] < baseline["class_iou"][1] + road_gain:
        failures.append("Road IoU gain is less than one percentage point")
    if candidate["mean_iou"] < baseline["mean_iou"]:
        failures.append("Five-class mean IoU regressed")
    for index in (0, 2, 3, 4):
        if candidate["class_iou"][index] < baseline["class_iou"][index] - nonroad_tolerance:
            failures.append(f"Non-road class {index} IoU regressed by more than one percentage point")
    if candidate["class_precision"][1] is None:
        failures.append("Road precision is undefined because no roads were predicted")
    elif candidate["class_precision"][1] < baseline["class_precision"][1]:
        failures.append("Road precision regressed")
    return {"passed": not failures, "failures": failures,
            "minimum_absolute_road_iou_gain": road_gain,
            "maximum_absolute_nonroad_iou_regression": nonroad_tolerance}
