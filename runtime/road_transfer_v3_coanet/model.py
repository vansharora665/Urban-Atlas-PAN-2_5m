"""Run inspected author architecture definitions with ordinary evaluation BN.

CoANet source: https://github.com/mj129/CoANet, revision b680c4f2.
Upstream GPL-3.0 license is retained in upstream/CoANet-main/LICENSE; its README
also restricts use to non-commercial research. This experiment is not deployed.
No upstream training script or automatic ImageNet download is executed.
"""

import ast
from collections import OrderedDict
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from road_network_v1.audit import digest


UPSTREAM = Path(__file__).parent/"upstream/CoANet-main/modeling"
SOURCE_HASHES = {
    "backbone/resnet.py": "fc90bcf7546090d452117766d7811bb29b1e7fbd437aba9a99cd0d56a77fd80c",
    "aspp.py": "60a0602142d641396ba6e088ee706fe8d021c86152a70e75a585681234879bd0",
    "decoder.py": "45db8df1060e188f062456db4fb660dd932bb477a736797f9d0d1537a4219f7b",
    "connect.py": "e69505931441c028cf02087496cc0fd66999fccea68f40b30dc94d29751f8508",
    "coanet.py": "73c79d134d0fc7772d90ff68b0c125a1afe52ce280fcfc03a4484db2a3236bd2"}


def author_model():
    namespace = {"torch": torch, "nn": nn, "F": torch.nn.functional, "math": math,
                 "SynchronizedBatchNorm2d": nn.BatchNorm2d}
    for name in SOURCE_HASHES:
        path = UPSTREAM/name
        if digest(path) != SOURCE_HASHES[name]:
            raise ValueError(f"Inspected architecture source changed: {name}")
        tree = ast.parse(path.read_text())
        definitions = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
        if name == "backbone/resnet.py":
            # The checkpoint supplies the full trained backbone. Disable the
            # old constructor's redundant, automatic ImageNet network fetch.
            namespace["build_backbone"] = lambda backbone, output_stride, BatchNorm: namespace["ResNet101"](
                output_stride, BatchNorm, pretrained=False)
    return namespace["CoANet"](num_classes=1, num_neighbor=9, output_stride=8, sync_bn=False)


def load_checkpoint(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    state = saved.get("state_dict", saved)
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Expected a tensor-only state dictionary")
    state = OrderedDict((key.removeprefix("module."), value) for key, value in state.items())
    model = author_model()
    model.load_state_dict(state, strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Non-finite model tensors")
    return model, {"safe_weights_only": True, "strict_load": True,
                   "saved_tensors": len(state), "parameters": sum(p.numel() for p in model.parameters()),
                   "source_revision": "b680c4f2b68e06a5ada03c63441a179fd3244490"}


def prepare_input(image, gamma=1.):
    image = np.asarray(image, np.float32)
    if image.ndim != 3 or image.shape[0] != 4 or not np.isfinite(image).all():
        raise ValueError("Expected finite, channels-first B/G/R/NIR")
    if image.min() < 0 or image.max() > 1 or not .4 <= gamma <= 2.:
        raise ValueError("Expected normalized [0,1] values and bounded gamma")
    rgb = image[[2, 1, 0]]**gamma
    mean = np.array([.485, .456, .406], np.float32)[:, None, None]
    std = np.array([.229, .224, .225], np.float32)[:, None, None]
    return np.ascontiguousarray((rgb-mean)/std)


def configure_training(model, thaw=False):
    model.train()
    for name, parameter in model.named_parameters():
        encoder = name.startswith("backbone.")
        parameter.requires_grad_(not encoder or (thaw and name.startswith(("backbone.layer3.", "backbone.layer4."))))
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def forward_logits(model, image):
    """Expose the original segmentation head's pre-sigmoid logits for stable BCE."""
    e1, e2, e3, e4 = model.backbone(image)
    decoded = model.decoder(e1, e2, e3, model.aspp(e4))
    connection = model.connect
    return (connection.seg_branch(decoded), connection.se(connection.connect_branch(decoded)),
            connection.se_d1(connection.connect_branch_d1(decoded)))
