import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from open_clip import create_model_and_transforms, tokenizer
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

import pandas as pd

from IPython.display import Image

from PIL import Image
from open_clip import create_model_and_transforms, tokenizer
import torch.nn.functional as F
import open_clip

def image_to_image(image_path, model, preprocess, df):
    # This converts the image to a tensor
    image = preprocess(Image.open(image_path)).unsqueeze(0)

    # This calculates the query embedding
    query_embedding = F.normalize(model.encode_image(image))

    impath = 'coco_images_resized/' + df.loc[(df['embedding'].apply(lambda x: F.cosine_similarity(query_embedding, torch.tensor(x).unsqueeze(0)).item())).idxmax()]['file_name']
    return impath

def text_to_image(query, model, df):
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.eval()
    text = tokenizer([query])
    query_embedding = F.normalize(model.encode_text(text))

    # Retrieve the image path that corresponds to the embedding in `df`
    # with the highest cosine similarity to query_embedding
    impath =  'coco_images_resized/' + df.loc[(df['embedding'].apply(lambda x: F.cosine_similarity(query_embedding, torch.tensor(x).unsqueeze(0)).item())).idxmax()]['file_name']
    return impath

def hybrid_to_image(query, image_path, lam, model, preprocess, df):
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    image_query = F.normalize(model.encode_image(image))
    text = tokenizer([query])
    text_query = F.normalize(model.encode_text(text))

    query = F.normalize(lam * text_query + (1.0 - lam) * image_query)

    impath = 'coco_images_resized/' + df.loc[(df['embedding'].apply(lambda x: F.cosine_similarity(query, torch.tensor(x).unsqueeze(0)).item())).idxmax()]['file_name']
    return impath