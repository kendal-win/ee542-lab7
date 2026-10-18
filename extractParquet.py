import os
import pandas as pd
from PIL import Image
import io
import time

os.makedirs('data/train/cat', exist_ok=True)
os.makedirs('data/train/dog', exist_ok=True)

startTime = time.time()

df = pd.read_parquet('train-00000-of-00002.parquet')
for i, row in df.iterrows():
    img_data = row['image']
    if isinstance(img_data, dict) and 'bytes' in img_data:
        img_bytes = img_data['bytes']
        img = Image.open(io.BytesIO(img_bytes))
    else:
        img = Image.open(io.BytesIO(img_data))
    label = row['labels'] 
    if label == 0:
        img.save(f'data/train/cat/cat_{i}.jpg')
    else:
        img.save(f'data/train/dog/dog_{i}.jpg')

df = pd.read_parquet('train-00001-of-00002.parquet')
for i, row in df.iterrows():
    img_data = row['image']
    if isinstance(img_data, dict) and 'bytes' in img_data:
        img_bytes = img_data['bytes']
        img = Image.open(io.BytesIO(img_bytes))
    else:
        img = Image.open(io.BytesIO(img_data))
    label = row['labels'] 
    if label == 0:
        img.save(f'data/train/cat/cat_0{i}.jpg')
    else:
        img.save(f'data/train/dog/dog_0{i}.jpg')

print(f"Time: {time.time() - startTime:.4f}") 