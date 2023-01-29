from flask import Flask, request, render_template
from PIL import Image
from io import BytesIO
import base64
from matplotlib import pyplot as plt
import cv2 
# necessary libraries for python
import cv2
import numpy as np
import matplotlib.image as mg
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras.utils import get_file
import IPython.display as display
import matplotlib as mpl
import numpy as np
import PIL.Image
import time
import functools
from keras.preprocessing import image
import os
from os import listdir
import keras.utils as image
from PIL import Image as PImage
from keras.applications.vgg19 import VGG19
from  keras.applications.vgg19 import decode_predictions
from keras.applications.vgg19 import preprocess_input

app = Flask(__name__,template_folder='templates',static_folder='statics')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']

        # Process the image here
        img = function1(file)
        img = Image.open(file)

        # Convert the image to a base64 encoded string
        img_io = BytesIO()
        img.save(img_io, 'JPEG', quality=70)
        img_io.seek(0)
        processed_img = base64.b64encode(img_io.getvalue()).decode()

        return render_template('display.html', processed_img=processed_img)
    return render_template('display.html')

def function1(img_path):
    img = plt.imread(img_path)
    dst = cv2.fastNlMeansDenoisingColored(img,None, 10, 10, 7, 21)
    return dst

def function2(img_path,clip_hist_percent=1):
    
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    
    
    
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    #configure CLAHE
    clahe = cv2.createCLAHE(clipLimit=0.5,tileGridSize=(8,8))

    #0 to 'L' channel, 1 to 'a' channel, and 2 to 'b' channel
    
#     ret, thresh3 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img[:,:,0] = clahe.apply(img[:,:,0])
    
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    
#     ret, thresh3 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

   
    j = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return j

# @app.route('/process_image', methods=['POST'])
# def process_image():
#     file = request.files['image']
#     function = request.form.get("function")
#     if function == "function1":
#         processed_image = function1(file)
#     elif function == "function2":
#         processed_image = function2(file)
#     # elif function == "function3":
#     #     processed_image = function3(file)
#     # elif function == "function4":
#     #     processed_image = function4(file)
#     # elif function == "function5":
#     #     processed_image = function5(file)
#     # Save the processed image
#     processed_image.save("processed.jpg")
#     return redirect(url_for('display'))

if __name__ == '__main__':
    app.run(debug=True)
