import os

import omrmodules
import torch
import cv2 as cv

def main():

    ###
    # preparation
    IMAGE = os.path.join('samples', 'aural_tests.jpg')

    # load models
    MODEL_MEASURE = os.path.join('saved_models', 'muscima_measures.pt')
    MODEL_OBJECT = os.path.join('saved_models', 'muscima_objects_pitchonly.pt')
    model_measures = torch.load(MODEL_MEASURE, map_location=torch.device('cpu'))
    model_objects = torch.load(MODEL_OBJECT, map_location=torch.device('cpu'))
    model_measures.eval()
    model_objects.eval()
    model_measures([torch.rand(1,400,400)])
    model_objects([torch.rand(1,400,400)])
    
    ###
    #

    # read image
    image = cv.imread(IMAGE)

    # pre-process image
    image = omrmodules.normalization.preprocess.processnotesheet(image)

    # model inference


    # pass results to SongFactory

    # Write to JSON

    
    print('done')
    pass


if __name__ == "__main__":

    main()