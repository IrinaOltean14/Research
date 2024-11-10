import pandas as pd
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
import os


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


dataset_path = r"E:\Sem5\Licenta\Dataset\SemArt\SemArt\semart_train.csv"


data = pd.read_csv(dataset_path, sep='\t', encoding='ISO-8859-1')

# Example: Generate captions for the first few images in the dataset
for index, row in data.iterrows():
    if index >= 7:
        break
    image_file = row['IMAGE_FILE']
    image_path = os.path.join("E:/Sem5/Licenta/Dataset/SemArt/SemArt/Images", image_file)
    image = Image.open(image_path)


    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=30)

    caption = processor.decode(outputs[0], skip_special_tokens=True)
    print(f"Image: {image_file}, Caption: {caption}")