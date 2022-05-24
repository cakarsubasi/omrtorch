import os

import omrmodules
import torch
import cv2 as cv
import glob
from PIL import Image

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
    imagepath = os.path.join('muscima/v2.0/data/images')
    imgs = sorted(glob.glob(os.path.join(imagepath, '*.png')))
    omrengine = OMREngine()
    imagenum = len(imgs)
    limit = imagenum

    for idx, img in enumerate(imgs):
        print(f"Processing {idx+1}/{limit}")
        sample_image = cv.imread(img)
        sample_image = np.average(sample_image, axis=2)
        measure_dict, object_dict = omrengine(sample_image)
        song = SongFactory(sample_image, measure_dict, object_dict).song

        filename = f"muscima_{idx}.json"
        with open(os.path.join("jsons", filename), "w") as wb:
            wb.write(song.toJSON())

        if idx == limit - 1:
            break
  
    
    
    #omrengine(sample_image)

    pass