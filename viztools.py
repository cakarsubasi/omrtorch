import torch
import numpy as np
from PIL import Image
from torchvision.utils import draw_bounding_boxes
import os

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


def show_preprocessed(x): 
    image = Image.fromarray(x)
    image = image.convert("RGB")
    return image

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

def save_images(prefix_string, im_preprocessed, image, measure_dict, object_dict, songFactory, OUTPUT_DIR):
    im_preprocessed = show_preprocessed(im_preprocessed)
    im_measures = show_measures(
        image, measure_dict, songFactory.MEASURE_THRESHOLD)
    im_noteheads = show_noteheads(
        image, object_dict, songFactory.OBJECT_THRESHOLD)
    im_segments = show_segments(image, songFactory.song)

    im_preprocessed.save(os.path.join(
        OUTPUT_DIR, f"{prefix_string}__preprocessed.jpg"))
    im_measures.save(os.path.join(
        OUTPUT_DIR, f"{prefix_string}_measures.jpg"))
    im_noteheads.save(os.path.join(
        OUTPUT_DIR, f"{prefix_string}_noteheads.jpg"))
    im_segments.save(os.path.join(
        OUTPUT_DIR, f"{prefix_string}_segments.jpg"))
    print("Saved images.")