# Classes and methods for handling Muscima datasets in PyTorch
import os
import json

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class MuscimaMeasures(Dataset):
  '''
  Class for constructing a dataset object for system measure,
  stave measure, and stave detection on the Muscima++ dataset
  '''
  def __init__(self, imagepath, annotationpath, transforms=None):
    self.imgpath = imagepath
    self.annpath = annotationpath

    self.imgs = list(sorted(os.listdir(imagepath)))
    self.anns = list(sorted(os.listdir(annotationpath)))

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
    
      objectlabels = ['system_measures', 'stave_measures', 'staves']
    
      for idx, label in enumerate(objectlabels):
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

