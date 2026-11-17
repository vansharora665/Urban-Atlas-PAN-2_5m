from __future__ import annotations

from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from torch import nn


class NativeAttentionUNetPP(nn.Module):
    def __init__(self, encoder_name="resnet34", in_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.network = smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights=None,
                                       decoder_attention_type="scse", in_channels=in_channels,
                                       classes=5, activation=None)
        self.dropout = nn.Dropout2d(.10)
        self.road_head = nn.Conv2d(16,1,1)
        self.boundary_head = nn.Conv2d(16,1,1)

    def decode_features(self, image):
        self.network.check_input_shape(image)
        features = self.network.encoder(image)
        return self.dropout(self.network.decoder(features))

    def forward(self, image):
        decoded = self.decode_features(image)
        return {"semantic":self.network.segmentation_head(decoded),
                "road":self.road_head(decoded),"boundary":self.boundary_head(decoded)}

    def initialize_encoder(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Encoder initialization is missing: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path,map_location="cpu",weights_only=True)
        if "encoder_state_dict" in checkpoint:
            if checkpoint.get("stage") != "self_supervised_masked_reconstruction" or not checkpoint.get("pretraining_complete"):
                raise ValueError("Expected a completed encoder-pretraining checkpoint")
            encoder = checkpoint["encoder_state_dict"]
        else:
            state = checkpoint.get("model",checkpoint.get("state_dict",checkpoint))
            encoder = {key.removeprefix("encoder."):value for key,value in state.items() if key.startswith("encoder.")}
        if not encoder:
            raise ValueError("Checkpoint contains no compatible encoder parameters")
        conversion = None
        expected = self.network.encoder.state_dict()["conv1.weight"]
        old = encoder.get("conv1.weight")
        if old is not None and old.shape != expected.shape:
            if old.shape[1] != 1 or expected.shape[1] != 4 or old.shape[0] != expected.shape[0]:
                raise ValueError("Unsupported first-convolution channel conversion")
            encoder["conv1.weight"] = old.repeat(1,4,1,1)/4
            conversion = "Single-channel conv1 repeated across B/G/R/NIR and divided by four; not spectral pretraining"
        self.network.encoder.load_state_dict(encoder,strict=True)
        return {"path":str(checkpoint_path.resolve()),"scope":"encoder_only",
                "tensor_count":len(encoder),"input_channels":self.in_channels,
                "first_conv_conversion":conversion,
                "note":"Existing project encoder trained with weak labels; fresh attention decoder and auxiliary heads require reviewed native-resolution supervision"}


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
