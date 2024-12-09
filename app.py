from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import numpy as np
from io import BytesIO
from IPython.display import Image
from PIL import Image
from open_clip import create_model_and_transforms, tokenizer
import torch.nn.functional as F
import open_clip
import torch
import torchvision.transforms as transforms
import pandas as pd
from tqdm import tqdm
from image_search import image_to_image, text_to_image, hybrid_to_image

app = Flask(__name__)

# Path to save uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compress', methods=['POST'])
def compress():
    file = request.files['file']
    query = request.form['query']
    lam = int(request.form['lam'])

    df = pd.read_pickle('image_embeddings.pickle')
    model, _, preprocess = create_model_and_transforms('ViT-B/32', pretrained='openai')
    
    '''
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Load the image
        image_np = load_image(file_path)
        
        # Compress the image
        compressed_image_np = image_compression_svd(image_np, rank)
        
        # Save the compressed image
        compressed_image = Image.fromarray(compressed_image_np)
        original_image = Image.fromarray(image_np)
        width, height = original_image.size

        combined_image = Image.new('RGB', (width * 2, height))
        
        # Paste original and quantized images side by side
        combined_image.paste(original_image, (0, 0))
        combined_image.paste(compressed_image, (width, 0))

        compressed_image_io = BytesIO()
        combined_image.save(compressed_image_io, format='PNG')
        compressed_image_io.seek(0)

        # Send the compressed image back to the front-end
    '''
    
    im = image_to_image(secure_filename(file.filename), model, preprocess, df)

    return send_file(im, mimetype='image/jpg')

if __name__ == '__main__':
    app.run(port=3000, debug=True)