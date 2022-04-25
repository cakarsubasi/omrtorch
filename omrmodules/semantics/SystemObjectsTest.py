import numpy as np
import os
import pickle
import torch
import cv2 as cv

import SystemObjects

def main(root: str):

      IMAGE_PATH = os.path.join(root, "samples/sample_image.png")
      MEASURE_PATH = os.path.join(root, "samples/measuredetections.dict")
      OBJECT_PATH = os.path.join(root, "samples/objects_gt.dict")

      image = cv.imread(IMAGE_PATH)

      with open(MEASURE_PATH, "rb") as measure_dict:
         measures = torch.load(measure_dict, map_location=torch.device('cpu'))
      with open(OBJECT_PATH, "rb") as object_dict:
         objects = torch.load(object_dict, map_location=torch.device('cpu'))
      
      factory = SystemObjects.SongFactory(image, measures, objects)

      process_measuresTest()
      sortTest(factory)

      print("All assertions passed.")

def sortTest(factory):
   assert (factory.song.systems[0] < factory.song.systems[1])

   print("sort passed")
   pass

def process_measuresTest():
      some_measures = np.array([[0.59070754, 0.37419364, 0.7296071 , 0.46695033],
         [0.05971682, 0.37450802, 0.25843462, 0.46590468],
         [0.420092  , 0.374791  , 0.58300257, 0.4671924 ],
         [0.25922823, 0.37479445, 0.41415334, 0.46614403],
         [0.7334418 , 0.37778202, 0.8635829 , 0.46550924]],)

      some_measures = SystemObjects.process_measures(some_measures)

      assert (np.all(some_measures[1:,0] == some_measures[:-1,2]))

      print("process_measures passed")

if __name__ == "__main__":
      main("semantics")
      