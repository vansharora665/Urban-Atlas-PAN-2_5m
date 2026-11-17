"""Trained five-class SegFormer V9 core, without file-based hybrid refinements."""

from torch import nn
from torch.nn import functional as F
import segmentation_models_pytorch as smp


class DeploymentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = smp.Segformer(
            encoder_name="mit_b0", encoder_weights=None, in_channels=1, classes=5)

    def source_logits(self, image):
        height, width = image.shape[-2:]
        image = F.pad(image, (0, (32 - width % 32) % 32,
                              0, (32 - height % 32) % 32), mode="replicate")
        return self.network(image)[..., :height, :width]

    def forward(self, image):
        # Original training order: water, roads, built-up, barren, vegetation.
        return self.source_logits(image)[:, [0, 4, 3, 2, 1]]
