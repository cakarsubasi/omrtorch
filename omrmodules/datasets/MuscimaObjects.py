# Classes and methods for handling Muscima datasets in PyTorch
import os
import xmlschema
import glob
import re

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from ..visionutils import transforms as T

__pitch_objects__ = ['noteheadFull', 'noteheadHalf', 'noteheadWhole', 'accidentalSharp', 'accidentalFlat', 'accidentalNatural',
                    'gCflef', 'fClef', 'cClef']

class MuscimaObjects(Dataset):
  def __init__(self, root, label_list=None, transforms=None):
    # TODO add label list handling
    self.root = root
    
    imagepath = os.path.join(root, 'v2.0/data/images')
    annotationpath = os.path.join(root, 'v2.0/data/annotations')
    classschema = os.path.join(root, 'v2.0/specifications/NodeClasses_Schema.xsd')
    classfile = os.path.join(root, 'v2.0/specifications/mff-muscima-mlclasses-annot.xml')

    self.anns = sorted(glob.glob(os.path.join(annotationpath, '*.xml')))
    self.imgs = sorted(glob.glob(os.path.join(imagepath, '*.png')))
    schema = glob.glob(os.path.join(annotationpath, '*.xsd'))
    self.xs = xmlschema.XMLSchema(schema)

    self.label_list = getObjectLabels(classfile, classschema) if label_list is None else label_list

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
      if classname not in self.label_list:
        continue

      xmin = node['Left']
      ymin = node['Top']
      width = node['Width']
      height = node['Height']
      boxes.append([xmin, ymin, xmin+width, ymin+height])
      labels.append(self.label_list.index(classname)+1)
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


def getObjectLabels(classfile, schema):

  label_dict_tree = getMuscimaClassDict(classfile, schema)
  label_list = []
  for category in label_dict_tree:
    for element in label_dict_tree[category]:
      label_list.append(element)
  return label_list

def getMuscimaClassDict(classfile, schema,
                        ignored_categories=['layout', 'misc', 'notation', 'notations', 'special', 'text']):
  # get grouped label dictionary for all classes
  xs_classes = xmlschema.XMLSchema(schema)
  classes = xs_classes.to_dict(classfile)

  label_dict_tree = {}
  pat = re.compile(r"/")
  for idx, glyph in enumerate(classes['NodeClass']):
    tree = pat.split(glyph['GroupName'])
    if tree[0] not in label_dict_tree:
      label_dict_tree[tree[0]] = []
    label_dict_tree[tree[0]].append(tree[1])

  for element in ignored_categories:
    label_dict_tree.pop(element)

  return label_dict_tree

def get_transform(train):
  transforms = []
  # converts the image, a PIL image, into a PyTorch Tensor
  transforms.append(T.ToTensor())
  if train:
      # during training, randomly flip the training images
      # and ground-truth for data augmentation
      transforms.append(T.RandomHorizontalFlip(0.5))
  return T.Compose(transforms)