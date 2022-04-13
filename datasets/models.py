import torch
import torchvision
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
    rpn_pre_nms_top_n_train (2000)
    rpn_pre_nms_top_n_test  (1000)
    rpn_post_nms_top_n_train (2000)
    rpn_post_nms_top_n_test  (1000)
    rpn_nms_thresh (0.7)
    rpn_fg_iou_thresh (0.7)
    rpn_bg_iou_thresh (0.3)
    rpn_batch_size_per_image (256)
    rpn_score_thresh (0.0)

    box_score_thresh (0.05)
    box_nms_thresh (0.5)
    box_detections_per_img
    box_fg_iou_thresh (0.5)
    box_bg_iou_thresh (0.5)
    '''
    assert backbone in __backbones__

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
        box_detections_per_img=1000,
        rpn_batch_size_per_image=512,
        rpn_fg_iou_thresh=0.6,
        rpn_bg_iou_thresh=0.4,)

    return model

def _get_conv_next(num_classes=4):
    backbone = torchvision.models.convnext_small(pretrained=True).features

    backbone.out_channels = 768
    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    aspects = (0.5, 1.0, 2.0)
    
    # Option 1
    anchor_sizes = ((32, 64, 128, 256, 512),)
    aspect_ratios = ((0.5, 1.0, 2.0),)
    rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    
    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be ['0']. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    
    # Option 1
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)
    
    
    
    #rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=rpn_anchor_generator,
                       box_roi_pool=roi_pooler,
                       box_detections_per_img=1000)

    return model

def _get_conv_next2(num_classes=4):
    backbone = torchvision.models.convnext_small(pretrained=True).features

    backbone.out_channels = 768
    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    aspects = (0.5, 1.0, 2.0)
    
    # Option 2
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = (aspects,) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    
    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be ['0']. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3', '4'],
                                                    output_size=7,
                                                    sampling_ratio=2)
    
    
    rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=rpn_anchor_generator,
                       rpn_head = rpn_head,
                       box_roi_pool=roi_pooler,
                       box_detections_per_img=1000)

    return model

def _get_faster_rcnn_with_n_classes(n: int):
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, trainable_backbone_layers=5)

  in_features = model.roi_heads.box_predictor.cls_score.in_features

  model.roi_heads.detections_per_img = 400

  aspects = (0.38, 0.75, 1.14)
  anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
  aspect_ratios = (aspects,) * len(anchor_sizes)
  rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

  model.rpn.anchor_generator = rpn_anchor_generator

  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n)

  return model