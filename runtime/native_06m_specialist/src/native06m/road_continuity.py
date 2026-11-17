"""Road-only native-resolution correction on a frozen joint land-cover model."""

import torch
from copy import deepcopy
from torch.nn import functional as F

from .joint_refinement import JointNativeRefinement, road_hard_negative_loss
from .road_detail import RoadDetailHead,road_margin


class NativeRoadContinuity(JointNativeRefinement):
    """Non-road logits and every existing backbone parameter remain unchanged."""

    def __init__(self,encoder_name="resnet34",in_channels=4,residual_bound=6.,correction_scale=1.,head_mode="residual",ensemble_weight=.3,
                 protected_base_classes=(), protection_confidence=.75, road_margin_bias=0.):
        super().__init__(encoder_name,in_channels)
        self.requires_grad_(False)
        self.continuity_head=RoadDetailHead(in_channels,residual_bound)
        self.correction_scale=correction_scale
        if head_mode not in {"residual","independent","ensemble"}:raise ValueError("Unknown native road-head mode")
        self.head_mode=head_mode
        self.ensemble_weight=ensemble_weight
        self.protected_base_classes=tuple(protected_base_classes)
        self.protection_confidence=protection_confidence
        self.road_margin_bias=float(road_margin_bias)
        if any(c not in (0,2,3,4) for c in self.protected_base_classes):
            raise ValueError("Only non-road base classes can be protected")
        if head_mode=="ensemble":self.early_continuity_head=deepcopy(self.continuity_head)

    def initialize_joint(self,state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        if unexpected or any(not k.startswith(("continuity_head.","early_continuity_head.")) for k in missing):
            raise ValueError("Road-only correction requires the matching joint land-cover checkpoint")

    def train(self,mode=True):
        super().train(False)
        self.continuity_head.train(mode)
        return self

    def frozen_features(self,image):
        with torch.no_grad():
            self.network.check_input_shape(image)
            features=self.network.encoder(image)
            decoded=self.network.decoder(features)
            detail=self.native_detail(image)
            residual=self.native_output(self.native_fusion(torch.cat((decoded,detail),dim=1)))
            baseline=self.network.segmentation_head(decoded)+residual
            boundary=self.boundary_head(decoded)
        return features[1],decoded,baseline,boundary

    def forward(self,image):
        stem,decoded,baseline,boundary=self.frozen_features(image)
        output=self.continuity_head(image,stem,decoded,baseline)
        if self.head_mode in {"independent","ensemble"}:
            margin=self.correction_scale*output["road_delta"]-2.5
            if self.head_mode=="ensemble":
                early=self.early_continuity_head(image,stem,decoded,baseline)["road_delta"]-2.5
                probability=(1-self.ensemble_weight)*margin.sigmoid()+self.ensemble_weight*early.sigmoid()
                margin=torch.logit(probability.clamp(1e-6,1-1e-6))
            semantic=baseline.clone()
            semantic[:,1:2]=torch.logsumexp(baseline[:,[0,2,3,4]],1,keepdim=True)+margin
            output.update(semantic=semantic,road=margin,
                          road_delta=semantic[:,1:2]-baseline[:,1:2],correction_penalty=margin)
            output.update(base_semantic=baseline,boundary=boundary)
            return self.protect(output,baseline)
        if self.correction_scale!=1.:
            delta=output["road_delta"]*self.correction_scale
            semantic=baseline.clone()
            semantic[:,1:2]+=delta
            output.update(semantic=semantic,road_delta=delta,road=road_margin(semantic))
        output.update(base_semantic=baseline,boundary=boundary)
        return self.protect(output,baseline)

    def protect(self, output, baseline):
        bias = getattr(self, "road_margin_bias", 0.)
        if not self.protected_base_classes and bias == 0:
            return output
        probability = baseline.softmax(1)
        confidence, dominant = probability.max(1, keepdim=True)
        protected = torch.zeros_like(dominant, dtype=torch.bool)
        for class_id in self.protected_base_classes:
            protected |= dominant == class_id
        protected &= confidence >= self.protection_confidence
        semantic = output["semantic"].clone()
        semantic[:,1:2] = semantic[:,1:2]+bias
        # A road-only update cannot override confidently identified water.
        semantic[:,1:2] = torch.where(protected, baseline[:,1:2], semantic[:,1:2])
        output.update(semantic=semantic, road=road_margin(semantic))
        return output


def soft_skeleton(image,iterations=10):
    def erode(value):
        return torch.minimum(-F.max_pool2d(-value,(3,1),1,(1,0)),
                             -F.max_pool2d(-value,(1,3),1,(0,1)))
    def opening(value):
        return F.max_pool2d(erode(value),3,1,1)
    skeleton=F.relu(image-opening(image))
    for _ in range(iterations):
        image=erode(image)
        delta=F.relu(image-opening(image))
        skeleton=skeleton+F.relu(delta-skeleton*delta)
    return skeleton


def reviewed_continuity_loss(output,target):
    valid=target!=255
    positive=target==1
    if not valid.any():
        return output["road"].sum()*0
    margin=output["road"]
    hard=road_hard_negative_loss(margin,target,negative_ratio=4)
    p=margin[:,0].sigmoid()*valid
    y=positive.float()
    dice=margin.sum()*0
    topology=margin.sum()*0
    if positive.any():
        intersection=(p*y).sum()
        false_positive=(p*(~positive)*valid).sum()
        false_negative=((1-p)*y).sum()
        dice=1-(intersection+1)/(intersection+.4*false_positive+.6*false_negative+1)
        ps=soft_skeleton(p[:,None])
        ys=soft_skeleton(y[:,None])
        precision=(ps*y[:,None]).sum()/(ps.sum()+1e-5)
        recall=(ys*p[:,None]).sum()/(ys.sum()+1e-5)
        topology=1-(2*precision*recall+1e-5)/(precision+recall+1e-5)
    penalty=output.get("correction_penalty",output["road_delta"])[:,0][valid].square().mean()
    return hard+.4*dice+.3*topology+.005*penalty
