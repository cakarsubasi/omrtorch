from email.mime import image
import io
import json
import os
import flask
import werkzeug
import omrmodules
import torch
import cv2 as cv
import scipy.misc
import numpy as np
from PIL import Image
from flask import Flask, request

#curl -X POST -F "file=@/Users/abdullahkucuk/input_pic.jpg" http://localhost:5000/ for send input from terminal
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
print("preinitilization complete.")


def transform_image(image_bytes):
    decoded = cv.imdecode(np.frombuffer(image_bytes, np.uint8), 1)
    #decoded = cv.imdecode(image_bytes, cv.IMREAD_IGNORE_ORIENTATION)
    if (decoded.shape[1] > decoded.shape[0]):
        decoded = np.rot90(decoded, k=1, axes=(0,1))
        print("image rotated")
    image = omrmodules.normalization.preprocess.processnotesheet(decoded)
    image = (np.expand_dims(image, 0) / 255.0).astype(np.float32)
    image = [torch.from_numpy(image).to(device)]
    return image

@app.route('/', methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = flask.request.files['image0']
        filename = werkzeug.utils.secure_filename(file.filename)
        print("\nReceived image File name : " + file.filename)
        file.save(filename)
        f = open(filename, 'rb')

        img_bytes =  f.read()
        image = transform_image(img_bytes)
        print("transform complete")
        measure_dict = model_measures(image)
        print("measures detected")
        object_dict = model_objects(image)
        print("objects detected")
        songFactory = omrmodules.semantics.SystemObjects.SongFactory(image[0], measure_dict[0], object_dict[0])
        songstring = songFactory.song.toJSON()
        print("song constructed")
        with open("song.json", "w") as wb:
            wb.write(songstring)
        print('done')
        pass
    return songstring 


if __name__ == '__main__':
    app.run()
