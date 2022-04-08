# Classes and methods for handling Muscima datasets in PyTorch
import os
import xmlschema
import glob

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class MuscimaObjects(Dataset):
  def __init__(self, root, label_list=None, transforms=None):
    # TODO add label list handling
    self.root = root
    
    imagepath = os.path.join(root, 'v2.0/data/images')
    annotationpath = os.path.join(root, 'v2.0/data/annotations')
    classesfile = os.path.join(root, 'v2.0/specifications/NodeClasses_Schema.xsd')

    self.anns = sorted(glob.glob(os.path.join(annotationpath, '*.xml')))
    self.imgs = sorted(glob.glob(os.path.join(imagepath, '*.png')))
    schema = glob.glob(os.path.join(annotationpath, '*.xsd'))
    self.xs = xmlschema.XMLSchema(schema)

    self.transforms = transforms
    

  def __len__(self):
    return len(self.imgs)

  def __getitem__(self, idx):
    image = Image.open(self.imgs[idx])
    target = {}
    # retrieve the xml file
    nodes = self.xs.to_dict(self.anns[idx])
    labels = []
    boxes = []
    iscrowd = []
    image_id = torch.tensor([idx])
    for node in nodes['Node']:
      classname = node['ClassName']
      if classname not in label_list:
        continue

      xmin = node['Left']
      ymin = node['Top']
      width = node['Width']
      height = node['Height']
      boxes.append([xmin, ymin, xmin+width, ymin+height])
      # Hardcoded from the label list
      labels.append(label_list.index(classname)+1)
      # todo (optional), add masks
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    iscrowd = torch.zeros(len(labels), dtype=torch.int64)
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

    target['boxes'] = boxes
    target['labels'] = labels
    target['image_id'] = image_id
    target['area'] = area
    target['iscrowd'] = iscrowd

    if self.transforms is not None:
      image, target = self.transforms(image, target)

    return image, target