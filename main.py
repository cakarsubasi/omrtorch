import os

import omrmodules
import torch
import cv2 as cv
import pickle

import numpy as np

def main():

    ###
    # preparation
    IMAGE = os.path.join('samples', 'aural_tests.jpg')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load models
    MODEL_MEASURE = os.path.join('saved_models', 'muscima_measures.pt')
    MODEL_OBJECT = os.path.join('saved_models', 'muscima_objects_pitchonly.pt')
    model_measures = torch.load(MODEL_MEASURE, map_location=torch.device('cpu'))
    model_objects = torch.load(MODEL_OBJECT, map_location=torch.device('cpu'))
    model_measures.to(device)
    model_objects.to(device)
    model_measures.eval()
    model_objects.eval()
    model_measures([torch.rand(1,400,400).to(device)])
    model_objects([torch.rand(1,400,400).to(device)])

    
    ###
    #

    # read image
    image = cv.imread(IMAGE)

    # pre-process image
    image = omrmodules.normalization.preprocess.processnotesheet(image)

    # model inference
    image = (np.expand_dims(image, 0) / 255.0).astype(np.float32)
    image = [torch.from_numpy(image).to(device)]
    measure_dict = model_measures(image)
    object_dict = model_objects(image)

    # pass results to SongFactory
    songFactory = omrmodules.semantics.SystemObjects.SongFactory(image[0], measure_dict[0], object_dict[0])

    # Write to JSON
    songstring = songFactory.song.toJSON()

    with open("song.json", "w") as wb:
        wb.write(songstring)

    with open("song.dictionary", "wb") as wb:
        pickle.dump(songFactory.song, wb)
    
    print('done')
    pass


if __name__ == "__main__":

    main()