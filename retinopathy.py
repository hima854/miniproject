#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, jsonify, render_template, render_template_string
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import cv2
from PIL import Image
import io

app = Flask(__name__)

# Load your deep learning model
model = load_model('final_model_resnet.h5')

# Define a function to preprocess input image
def preprocess_image(img):
    # Convert PIL image to numpy array
    img_array = np.array(img)

    # Resize image to match model input size
    resized_img = cv2.resize(img_array, (224, 224))

    # Preprocess the image
    img_preprocessed = preprocess_input(resized_img)
    return img_preprocessed

# Function to enhance images
def enhance_image(image):
    # Your image enhancement code here
    
    return image

# Define a route to render the HTML form
@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)

@app.route('/predict', methods=['POST'])
def predict():
    if 'input-image' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['input-image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        img = Image.open(io.BytesIO(file.read()))
        enhanced_img = enhance_image(img)
        processed_img = preprocess_image(enhanced_img)
        prediction = model.predict(np.expand_dims(processed_img, axis=0))
        predicted_class_index = np.argmax(prediction)
        highlighted_boxes = ['class1', 'class2', 'class3', 'class4', 'class5']
        highlighted_box = f'class{predicted_class_index + 1}'
        return render_template('index.html', highlighted_box=highlighted_box)

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:





# In[ ]:




