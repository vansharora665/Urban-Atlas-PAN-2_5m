from __future__ import annotations

import torch
from torch.nn import functional as F


def boundaries(target):
    valid = target!=255
    edge = torch.zeros_like(valid)
    vertical = valid[:,1:] & valid[:,:-1] & (target[:,1:]!=target[:,:-1])
    horizontal = valid[:,:,1:] & valid[:,:,:-1] & (target[:,:,1:]!=target[:,:,:-1])
    edge[:,1:] |= vertical
    edge[:,:-1] |= vertical
    edge[:,:,1:] |= horizontal
    edge[:,:,:-1] |= horizontal
    safe = valid & (F.max_pool2d((~valid).float().unsqueeze(1),3,1,1).squeeze(1)==0)
    return edge.float(),safe


def soft_skeleton(image, iterations=5):
    def erode(x):
        return torch.minimum(-F.max_pool2d(-x,(3,1),1,(1,0)),
                             -F.max_pool2d(-x,(1,3),1,(0,1)))
    def opened(x):
        return F.max_pool2d(erode(x),3,1,1)
    skeleton = F.relu(image-opened(image))
    for _ in range(iterations):
        image = erode(image)
        delta = F.relu(image-opened(image))
        skeleton = skeleton + F.relu(delta-skeleton*delta)
    return skeleton


def topology_safe_region(valid, iterations=5):
    # Each erosion expands support by one; opening adds two more pixels.
    radius = iterations+2
    unknown = F.pad((~valid).float().unsqueeze(1),(radius,)*4,value=1)
    return F.max_pool2d(unknown,2*radius+1,1)==0


def topology_loss(probability, truth, valid, iterations=5):
    safe = topology_safe_region(valid,iterations).to(probability.dtype)
    p = probability*valid.unsqueeze(1)
    t = truth.unsqueeze(1).float()*valid.unsqueeze(1)
    true_skeleton = soft_skeleton(t,iterations)*safe
    if not torch.any(true_skeleton):
        return probability.sum()*0
    predicted_skeleton = soft_skeleton(p,iterations)*safe
    precision = ((predicted_skeleton*t).sum()+1e-6)/(predicted_skeleton.sum()+1e-6)
    sensitivity = ((true_skeleton*p).sum()+1e-6)/(true_skeleton.sum()+1e-6)
    return 1-(2*precision*sensitivity+1e-6)/(precision+sensitivity+1e-6)


def segmentation_loss(output, target, class_weights, config):
    logits = output["semantic"]
    valid = target!=255
    if not valid.any():
        return sum(value.sum()*0 for value in output.values())
    ce = F.cross_entropy(logits,target,weight=class_weights,ignore_index=255)
    probability = torch.softmax(logits,1)
    truth = F.one_hot(target.clamp_max(4),5).permute(0,3,1,2).float()*valid.unsqueeze(1)
    probability_valid = probability*valid.unsqueeze(1)
    intersection = (probability_valid*truth).sum((0,2,3))
    denominator = (probability_valid+truth).sum((0,2,3))
    present = truth.sum((0,2,3))>0
    dice = (1-(2*intersection+1e-5)/(denominator+1e-5))[present].mean()
    road_target = (target==1).float()
    road = F.binary_cross_entropy_with_logits(output["road"].squeeze(1)[valid],road_target[valid],
                pos_weight=torch.as_tensor(config.get("road_positive_weight",1.0),device=target.device))
    boundary = output["boundary"].sum()*0
    if config["boundary_weight"]>0:
        edge,safe = boundaries(target)
        if safe.any():
            boundary = F.binary_cross_entropy_with_logits(output["boundary"].squeeze(1)[safe],edge[safe])
    topology = (topology_loss(probability[:,1:2],target==1,valid)
                if config["topology_weight"]>0 else probability.sum()*0)
    return ce+config["dice_weight"]*dice+config["road_aux_weight"]*road+config["boundary_weight"]*boundary+config["topology_weight"]*topology


def confusion_metrics(matrix):
    import numpy as np
    matrix = np.asarray(matrix,dtype=np.int64)
    tp = np.diag(matrix)
    true = matrix.sum(1)
    predicted = matrix.sum(0)
    union = true+predicted-tp
    iou = [float(tp[i]/union[i]) if union[i] else None for i in range(5)]
    recall = [float(tp[i]/true[i]) if true[i] else None for i in range(5)]
    precision = [float(tp[i]/predicted[i]) if predicted[i] else None for i in range(5)]
    f1 = [float(2*tp[i]/(true[i]+predicted[i])) if true[i]+predicted[i] else None for i in range(5)]
    return {"confusion_matrix":matrix.tolist(),"evaluated_pixels":int(matrix.sum()),
            "pixel_accuracy":float(tp.sum()/matrix.sum()) if matrix.sum() else None,
            "mean_iou":float(np.mean([v for v in iou if v is not None])) if any(v is not None for v in iou) else None,
            "class_iou":iou,"class_recall":recall,"class_precision":precision,"class_f1":f1,
            "road_as_water_rate":float(matrix[1,0]/true[1]) if true[1] else None,
            "water_as_road_rate":float(matrix[0,1]/true[0]) if true[0] else None,
            "absent_class_policy":"Undefined class metrics are null, never fabricated as 0% or 100%"}
