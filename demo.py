from omrengine import OMREngine
from omrmodules.semantics import SystemObjects
from omrmodules.semantics.SystemObjects import SongFactory
import os
import cv2 as cv

if __name__ == '__main__':
    omrengine = OMREngine()
    IMAGE = os.path.join("samples", "demo_score.png")
    sample_image = cv.imread(IMAGE)
    sample_image = OMREngine.preprocess(sample_image)
    measure_dict, object_dict = omrengine(sample_image)
    song = SystemObjects.SongFactory(sample_image, measure_dict, object_dict).song
    song.toStream().show('musicxml')
    pass