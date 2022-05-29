import os

import omrmodules
import torch
import cv2 as cv
import glob
from PIL import Image
import pathlib
import viztools

from omrmodules.semantics.SystemObjects import SongFactory

import numpy as np

class OMREngine():
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        MODEL_MEASURE = os.path.join('saved_models', 'muscima_measures.pt')
        MODEL_OBJECT = os.path.join('saved_models', 'muscima_objects_pitchonly.pt')
        model_measures = torch.load(MODEL_MEASURE, map_location=torch.device('cpu'))
        model_objects = torch.load(MODEL_OBJECT, map_location=torch.device('cpu'))
        model_measures.to(self.device)
        model_objects.to(self.device)
        model_measures.eval()
        model_objects.eval()
        model_measures([torch.rand(1,400,400).to(self.device)])
        model_objects([torch.rand(1,400,400).to(self.device)])
        self.model_measures = model_measures
        self.model_objects = model_objects
    
    def __call__(self, image):
        if (len(image.shape)) == 2:
            image = np.expand_dims(image, 0)
        if image.dtype != np.float32 or np.max(image) > 1:
            image = (image / 255.0).astype(np.float32)

        image = [torch.from_numpy(image).to(self.device)]
        measure_dict = self.model_measures(image)
        object_dict = self.model_objects(image)
        return measure_dict[0], object_dict[0]

    def preprocess(image):
        image = omrmodules.normalization.preprocess.processnotesheet(image)
        image = np.expand_dims(image, 0)
        image = (image / 255.0).astype(np.float32)
        return image


# Test the extractor on muscima, the extractor should
# get through the entire dataset without crashing
if __name__ == '__main__':
    OUTPUT_DIR = "output_muscima"
    if not pathlib.Path(OUTPUT_DIR).exists():
        pathlib.Path(OUTPUT_DIR).mkdir()

    imagepath = os.path.join('muscima/v2.0/data/images')
    imgs = sorted(glob.glob(os.path.join(imagepath, '*.png')))
    torch.manual_seed(1)
    indices = torch.randperm(len(imgs)).tolist()
    imgs = [imgs[i] for i in indices[-40:]]
    paths = [pathlib.Path(path) for path in imgs]

    omrengine = OMREngine()
    imagenum = len(imgs)
    limit = imagenum

    
    for idx, path in enumerate(paths):
        print(f"Processing {idx+1}/{limit}")
        sample_image = cv.imread(path.__str__())
        sample_image = np.average(sample_image, axis=2)
        measure_dict, object_dict = omrengine(sample_image)
        factory = SongFactory(sample_image, measure_dict, object_dict)
        song = factory.song
        image = np.expand_dims(sample_image, 0)
        image = torch.asarray(image).type(torch.FloatTensor)
        if torch.max(image) > 2:
            image = image / 255.0
        filename = path.stem
        
        viztools.save_images(filename, sample_image, image, measure_dict, object_dict, factory, OUTPUT_DIR)
        with open(os.path.join(OUTPUT_DIR, f"{filename}.json"), "w") as wb:
            wb.write(song.toJSON())
        song_stream = factory.song.toStream()
        song_stream.write(fmt = 'musicxml', fp = os.path.join(OUTPUT_DIR, f"{filename}"))

        if idx == limit - 1:
            break
  

    pass