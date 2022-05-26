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
import viztools
import pathlib
import time
import threading

# curl -X POST -F "file=@/Users/abdullahkucuk/input_pic.jpg" http://localhost:5000/ for send input from terminal
app = Flask(__name__)
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("using CUDA")
else:
    device = torch.device('cpu')
    print("using CPU")
MODEL_MEASURE = os.path.join('saved_models', 'muscima_measures.pt')
MODEL_OBJECT = os.path.join('saved_models', 'muscima_objects_pitchonly.pt')
model_measures = torch.load(MODEL_MEASURE, map_location=torch.device('cpu'))
model_objects = torch.load(MODEL_OBJECT, map_location=torch.device('cpu'))
model_measures.to(device)
model_objects.to(device)
model_measures.eval()
model_objects.eval()
model_measures([torch.rand(1, 400, 400).to(device)])
model_objects([torch.rand(1, 400, 400).to(device)])
OUTPUT_DIR = "output"
if not pathlib.Path(OUTPUT_DIR).exists():
    pathlib.Path(OUTPUT_DIR).mkdir()
print("preinitilization complete.")


def transform_image(image_bytes):
    decoded = cv.imdecode(np.frombuffer(image_bytes, np.uint8), 1)
    image = omrmodules.normalization.preprocess.processnotesheet(decoded)
    return image


def convert_to_torch(preprocessed_image):
    image = (np.expand_dims(preprocessed_image, 0) / 255.0).astype(np.float32)
    image = [torch.from_numpy(image).to(device)]
    return image


def save_images(time_string, im_preprocessed, image, measure_dict, object_dict, songFactory):
    im_preprocessed = viztools.show_preprocessed(im_preprocessed)
    im_measures = viztools.show_measures(
        image, measure_dict, songFactory.MEASURE_THRESHOLD)
    im_noteheads = viztools.show_noteheads(
        image, object_dict, songFactory.OBJECT_THRESHOLD)
    im_segments = viztools.show_segments(image, songFactory.song)

    im_preprocessed.save(os.path.join(
        OUTPUT_DIR, f"{time_string}__preprocessed.jpg"))
    im_measures.save(os.path.join(
        OUTPUT_DIR, f"{time_string}_measures.jpg"))
    im_noteheads.save(os.path.join(
        OUTPUT_DIR, f"{time_string}_noteheads.jpg"))
    im_segments.save(os.path.join(
        OUTPUT_DIR, f"{time_string}_segments.jpg"))
    print("Saved images.")


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = flask.request.files['image0']
        named_tuple = time.localtime()
        time_string = time.strftime("%Y-%m-%d_%H.%M.%S", named_tuple)

        filename = werkzeug.utils.secure_filename(f"{time_string}___raw.jpg")
        print("\nReceived image File name : " + file.filename)
        file.save(os.path.join(OUTPUT_DIR, filename))
        f = open(os.path.join(OUTPUT_DIR, filename), 'rb')

        img_bytes = f.read()
        im_preprocessed = transform_image(img_bytes)
        image = convert_to_torch(im_preprocessed)
        print("transform complete")
        measure_dict = model_measures(image)
        print("measures detected")
        object_dict = model_objects(image)
        print("objects detected")
        songFactory = omrmodules.semantics.SystemObjects.SongFactory(
            image[0], measure_dict[0], object_dict[0], measure_threshold=0.50, object_threshold=0.50)
        songstring = songFactory.song.toJSON()
        print("song constructed")
        with open("song.json", "w") as wb:
            wb.write(songstring)

        # Save asynchronously
        save_output = threading.Thread(target = save_images,
            args=(time_string, im_preprocessed, image[0], measure_dict[0], object_dict[0], songFactory))
        save_output.start()

        print('done')

    return songstring


if __name__ == '__main__':
    app.run()
