import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import RPNHead
import torch.nn as nn

__backbones__ = ['resnet18', 'resnet34', 'resnet50',
         'resnet101', 'resnet152', 'resnext50_32x4d',
         'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2']

def muscima_fpn_model(
    num_classes=4,
    backbone='resnet50',
    aspects=(0.5, 1.0, 2.0)):
    '''
    TODO: 
    rpn_nms_thresh
    rpn_fg_iou_thresh
    rpn_bg_iou_thresh
    rpn_batch_size_per_image
    rpn_score_thresh

    box_score_thresh
    box_nms_thresh
    box_detections_per_img
    box_fg_iou_thresh
    box_bg_iou_thresh
    '''
    model_backbone = resnet_fpn_backbone(backbone, pretrained=True, trainable_layers=5)

    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = (aspects,) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    rpn_head = RPNHead(model_backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
    #box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead((backbone.out_channels, 7, 7 ), [256, 256, 256, 256])

    model = FasterRCNN(
        model_backbone, 
        num_classes=num_classes,
        rpn_anchor_generator=rpn_anchor_generator,
        rpn_head=rpn_head,
        box_detections_per_img=1000)

    return model