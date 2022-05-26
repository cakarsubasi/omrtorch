import torch
import numpy as np
from PIL import Image
from torchvision.utils import draw_bounding_boxes

from omrmodules.datasets.MuscimaObjects import __pitch_objects__
from omrmodules.semantics.SystemObjects import SongFactory, denormalize_bboxes
from omrmodules.datasets.MuscimaMeasures import visualize_bboxes

def visualize(image, boxes):
    viz_image = torch.asarray(image)
    viz_image = (viz_image*255).type(torch.ByteTensor)
    viz_boxes = denormalize_bboxes(np.asarray(boxes), viz_image)
    viz_boxes = torch.asarray(viz_boxes)

    viz_im_with_bounding_boxes = draw_bounding_boxes(
        viz_image, viz_boxes, width=5, colors="red")

    return Image.fromarray(np.moveaxis(viz_im_with_bounding_boxes.numpy(), 0, -1))


def show_preprocessed(x): return Image.fromarray(x)

def show_measures(image, measure_dict, threshold=0.75):
    return visualize_bboxes(image, measure_dict, threshold=threshold)

def show_noteheads(image, object_dict, threshold=0.0):
    label_dict = __pitch_objects__.copy()
    label_dict.insert(0, "__background__")
    return visualize_bboxes(image, object_dict, label_dict, threshold=threshold)

def show_segments(image, song):
    boxes = []
    for system in song.systems:
        for measure in system.measures:
            boxes.append(measure.bbox())
    return visualize(image, boxes)

def save_image(image : Image, name):
    image.save(name)