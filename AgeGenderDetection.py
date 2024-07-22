from flask import Flask, flash, request, redirect, url_for, render_template,Response,jsonify
import urllib.request
from flask_cors import CORS,cross_origin
import os
import io
# from numba import jit, cuda
from pathlib import Path
import cv2
import dlib
import numpy as np
import re
import argparse
from omegaconf import OmegaConf
from tensorflow.keras.utils import get_file
from src.factory import get_model
import base64
import json
import operator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import re
from collections import defaultdict
from functools import reduce
from io import BytesIO
from PIL import Image
from PIL import UnidentifiedImageError
import cv2
import numpy as np

app = Flask(__name__)
#CORS(app)

app.secret_key = "age_estimator"
pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/EfficientNetB3_224_weights.11-3.44.hdf5"
modhash = '6d7f7b7ced093a8b3ef6399163da6ece'

@app.route('/')
#@cross_origin()
def home():
    return render_template('index.html')

def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. weights.28-3.73.hdf5)")
    parser.add_argument("--margin", type=float, default=0.4,
                        help="margin around detected face for age-gender estimation")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="target image directory; if set, images in image_dir are used instead of webcam")
    args = parser.parse_args()
    return args
######################################MODEL######################################################################
args = get_args()
weight_file = args.weight_file
margin = args.margin
if not weight_file:
    weight_file = get_file("EfficientNetB3_224_weights.11-3.44.hdf5", pretrained_model,
                           cache_subdir="pretrained_models",
                           file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))
# for face detection
detector = dlib.get_frontal_face_detector()
print("after detection")
model_name, img_size = Path(weight_file).stem.split("_")[:2]
print("after model name")
img_size = int(img_size)
# print(img_size,"img_size")
cfg = OmegaConf.from_dotlist(["model.model_name=EfficientNetB3", "model.img_size=224"])
model = get_model(cfg)
model.load_weights(weight_file)
############################################################################################################
def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
    output_path = 'output_image.jpg'  # Replace with the path where you want to save the image
    cv2.imwrite(output_path, image)
# #####################################################################################################
def PIL_image_to_base64(pil_image):
    buffered = BytesIO()
    img1=pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue())


def base64_to_PIL_image(base64_img):
    return Image.open(BytesIO(base64.b64decode(base64_img)))
    
#@cross_origin()
def detect_and_draw(image):
    # try :
    #     image = base64_to_PIL_image(image_string)
    # except UnidentifiedImageError:
    #     return process()
    img = np.array(image)
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_img = cv2.flip(input_img, 1)
    #####################################################
    if img is not None:
        img_h, img_w, _ = np.shape(input_img)
        # detect faces using dlib detector
        detected = detector(input_img, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))
        # print(faces)
        # print(detected)
        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)
                cv2.rectangle(input_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                faces[i] = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1], (img_size, img_size))
            # predict ages and genders of the detected faces
            results = model.predict(faces)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()
            # draw results
            for i, d in enumerate(detected):
                label = "{}, {}".format(int(predicted_ages[i]),
                                        "M" if predicted_genders[i][0] < 0.5 else "F")
                draw_label(input_img, (d.left(), d.top()), label)
    img=input_img
        # imageRGB = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # imageRGB = input_img
    imageRGB = img[..., ::-1]
        # cv2.imwrite("test2.png", imageRGB)
    PIL_image = Image.fromarray(imageRGB)
    return PIL_image_to_base64(PIL_image)
###################################WEBCAM###################################################
# @app.route('/webcam')
# def index():
#     return render_template('layout.html')

@app.route('/process', methods=['POST'])
#@crossorigin
def process():
    # file1 = request.files['file1']
    # filestr1 = file1.read()
    # image_ascii = detect_and_draw(filestr1)
    # npimg1 = np.fromstring(filestr1, np.uint8)
        # convert numpy array to image
    # img1 = cv2.imdecode(npimg1, cv2.IMREAD_COLOR)
    if 'file' not in request.files:
        return "file not available"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
# 
    file = request.files['file']
    print(f"input image {file.filename}")
    decoded_image = Image.open(io.BytesIO(file.read()))
    decoded_image.save('decoded_image.jpg')
    #image_data = re.sub('^data:image/.+;base64,', '', input['img'])
    image_ascii = detect_and_draw(decoded_image)
    # response=jsonify({"item":[image_ascii]})
    # response.headers.add("Access-Control-Allow-Origin", "*")

    image_ascii=image_ascii.decode("utf-8")
    dictionary = {"image_ascii": image_ascii}
    json_object = json.dumps(dictionary)
    return json_object
    #return "Done"

###################################WEBCAM-END###################################################
if __name__ == "__main__":
    app.run(debug=True)
