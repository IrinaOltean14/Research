import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import torch
import numpy as np


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")


dataset_path = r"E:\Sem5\Licenta\Dataset\SemArt\SemArt\semart_train.csv"
data = pd.read_csv(dataset_path, sep='\t', encoding='ISO-8859-1')


candidate_captions = [
    "A beautiful landscape painting.",
    "A portrait of a person.",
    "An abstract piece of art.",
    "A depiction of a historical event.",
    "A serene still life.",
    "A vibrant city scene.",
    "A dramatic sunset.",
    "An intricate floral design."
]

for index, row in data.iterrows():
    if index >= 5:
        break
    image_file = row['IMAGE_FILE']
    image_path = os.path.join(r"E:\Sem5\Licenta\Dataset\SemArt\SemArt\Images", image_file)


    if not os.path.isfile(image_path):
        print(f"Image file not found: {image_path}")
        continue

    image = Image.open(image_path)

    inputs = processor(text=candidate_captions, images=image, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        image_embeddings = outputs.image_embeds
        text_embeddings = outputs.text_embeds

    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)  # Normalize
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)      # Normalize

    # Calculate similarity scores
    similarities = torch.matmul(image_embeddings, text_embeddings.T)  # Shape: [1, num_candidates]
    best_caption_idx = similarities.argmax().item()  # Get the index of the best matching caption

    # Output the best caption
    best_caption = candidate_captions[best_caption_idx]
    print(f"Image: {image_file}, Best Caption: {best_caption}")