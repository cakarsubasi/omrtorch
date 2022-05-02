import os

import omrmodules
import torch
import cv2 as cv
import pickle

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
        

def main():

    ###
    # preparation
    IMAGE = os.path.join('samples', 'aural_tests.jpg')

    # load models
    omrengine = OMREngine()
    
    ###
    #

    # read image
    image = cv.imread(IMAGE)

    # pre-process image
    image = OMREngine.preprocess(image)
    measure_dict, object_dict = omrengine(image)

    # pass results to SongFactory
    songFactory = omrmodules.semantics.SystemObjects.SongFactory(image, measure_dict, object_dict)

    # Write to JSON
    songstring = songFactory.song.toJSON()

    with open("song.json", "w") as wb:
        wb.write(songstring)

    with open("song.dictionary", "wb") as wb:
        pickle.dump(songFactory.song, wb)
    
    print('done')
    pass


if __name__ == "__main__":
    import argparse

    #parser = argparse.ArgumentParser(description='OMR Predictor')
    #parser.add_argument('file', type=str, help='File to predict')
    #parser.add_argument('--dst', type=str, default='./output', help='output directory')
#
    #args = parser.parse_args()

    main()