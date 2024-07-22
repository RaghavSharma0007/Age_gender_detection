import base64
import operator
import os
import re
from collections import defaultdict
from functools import reduce
from io import BytesIO
from random import choice, randint, shuffle
# import draw
from nltk import FreqDist
from PIL import Image, ImageChops, ImageOps
from PIL import UnidentifiedImageError
import cv2
import numpy as np



basedir = os.path.abspath(os.path.dirname(__file__))
# face_xml = os.path.join(basedir,'static','data', 'haarcascade_frontalface_alt.xml')
# eye_xml = os.path.join(basedir,'static','data', 'haarcascade_eye_tree_eyeglasses.xml')
# face_cascade = cv2.CascadeClassifier(face_xml)
# eye_cascade = cv2.CascadeClassifier(eye_xml)


#reference: https://stackoverflow.com/questions/31826335/how-to-convert-pil-image-image-object-to-base64-string





