# Classes and methods for handling Muscima datasets in PyTorch
import os
import json

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from ..visionutils import transforms as T
from torchvision.utils import draw_bounding_boxes


class MuscimaMeasures(Dataset):
  '''
  Class for constructing a dataset object for system measure,
  stave measure, and stave detection on the Muscima++ dataset
  '''
  def __init__(self, imagepath, annotationpath, label_list=None, transforms=None):
    self.imgpath = imagepath
    self.annpath = annotationpath

    self.imgs = list(sorted(os.listdir(imagepath)))
    self.anns = list(sorted(os.listdir(annotationpath)))
    self.label_list = getMeasureLabels() if label_list is None else label_list

    self.transforms = transforms

  def __len__(self):
    return len(self.imgs)

  def __getitem__(self, idx):
    image = Image.open(os.path.join(self.imgpath, self.imgs[idx]))
    target = {}
    height = 0
    width = 0
    labels = []
    boxes = []
    iscrowd = []
    image_id = torch.tensor([idx])
    with open(os.path.join(self.annpath, self.anns[idx])) as f:
      annotations = json.loads(f.read())
      width = annotations['width']
      height = annotations['height']
    
      for idx, label in enumerate(self.label_list):
        objs = annotations[label]
        for i in range(len(objs)):
          xbox = np.asarray([objs[i]['left'], objs[i]['right']])
          ybox = np.asarray([objs[i]['top'], objs[i]['bottom']])
          
          xmin = np.min(xbox)
          xmax = np.max(xbox)
          ymin = np.min(ybox)
          ymax = np.max(ybox)
          boxes.append([xmin, ymin, xmax, ymax])
          labels.append(idx+1)
          iscrowd.append(0)
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    
    target['boxes'] = boxes
    target['labels'] = labels
    target['image_id'] = image_id
    target['area'] = area
    target['iscrowd'] = iscrowd
    #target['width'] = width
    #target['height'] = height

    if self.transforms is not None:
      image, target = self.transforms(image, target)

    return image, target

def getMeasureLabels(system_measures=True, stave_measures=True, staves=False):
  '''
  Generates a class map for the measures dataset
  By default returns system measures and stave measures only
  system_measures: bounding boxes for system measures
  stave_measures: bounding boxes for staff measures
  staves: bounding boxes for whole staves
  '''
  label_map = []
  if system_measures:
    label_map.append('system_measures')
  if stave_measures:
    label_map.append('stave_measures')
  if staves:
    label_map.append('staves')
  return label_map

def getListofClassNames(labels, label_strs):
  '''
  Converts ordered list of numerical labels
  into ordered strings of labels for visualization purposes
  '''
  class_names = []
  for label in labels:
      class_names.append(label_strs[label-1])
  return class_names 

def get_transform(train):
  transforms = []
  # converts the image, a PIL image, into a PyTorch Tensor
  transforms.append(T.ToTensor())
  if train:
      # during training, randomly flip the training images
      # and ground-truth for data augmentation
      transforms.append(T.RandomHorizontalFlip(0.5))
  return T.Compose(transforms)

  
def visualize_bboxes(image, target, labels=None, threshold=0.5):
    if labels is None:
      labels = {1 : 'system_measures', 2: 'measures'}
    colors = ['000000', 'red', 'blue', 'green', 'yellow']
    sample_boxes = target['boxes']
    sample_scores = target['scores']
    sample_labels = target['labels']
    idx = torch.where(sample_scores > threshold)
    boxes_sliced = sample_boxes[idx]
    labels_sliced = sample_labels[idx].tolist()
    colors_list = None
    labels_list_str = None
    if labels is not None:
        labels_list_str = list(labels[val] for val in labels_sliced)
        colors_list = list(colors[val % len(colors)] for val in labels_sliced)
    if image.__class__ is np.ndarray:
      image = torch.tensor(image)
    sample_image = (image*255).type(torch.ByteTensor)

    sample_im_with_bounding_boxes = draw_bounding_boxes(sample_image, boxes_sliced, labels_list_str, colors=colors_list, width=3, font_size=32)

    return Image.fromarray(np.moveaxis(sample_im_with_bounding_boxes.numpy(), 0, -1))
