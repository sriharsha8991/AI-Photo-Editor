import os
from flask import Flask, request, render_template
from io import BytesIO
from PIL import Image
import base64
import cv2

app = Flask(__name__,template_folder='templates',static_folder='statics')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        # Save the uploaded file to a temporary location in the 'temp' directory
        temp_dir = 'temp_data'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        filename = os.path.join(temp_dir, file.filename)
        file.save(filename)
        # Convert the image to grayscale
        gray = function2(filename)
        # Convert the grayscale image back to an image and save it to a BytesIO object
        img = Image.fromarray(gray)
        img_io = BytesIO()
        img.save(img_io, 'JPEG', quality=70)
        img_io.seek(0)
        processed_img = base64.b64encode(img_io.getvalue()).decode()

        return render_template('display.html', processed_img=processed_img)
    return render_template('display.html')

def function1(filename):
    img = cv2.imread(filename)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

if __name__ == '__main__':
    app.run(debug=True)

