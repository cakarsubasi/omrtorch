from email.mime import image
import io
import json
import os

import omrmodules
import torch
import cv2 as cv

import numpy as np
from PIL import Image
from flask import Flask, request

#curl -X POST -F "file=@/Users/abdullahkucuk/input_pic.jpg" http://localhost:5000/predict for send input from terminal
app = Flask(__name__)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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


def transform_image(image_bytes):
    decoded = cv.imdecode(np.frombuffer(image_bytes, np.uint8),1)
    image = omrmodules.normalization.preprocess.processnotesheet(decoded)
    image = (np.expand_dims(image, 0) / 255.0).astype(np.float32)
    image = [torch.from_numpy(image).to(device)]
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        image = transform_image(img_bytes)
        measure_dict = model_measures(image)
        object_dict = model_objects(image)
        songFactory = omrmodules.semantics.SystemObjects.SongFactory(image[0], measure_dict[0], object_dict[0])
        songstring = songFactory.song.toJSON()
        with open("song.json", "w") as wb:
            wb.write(songstring)
        print('done')
        pass
    return 'done \n'    


if __name__ == '__main__':
    app.run()
