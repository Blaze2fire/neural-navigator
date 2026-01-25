"""
Neural Navigator Data Loader
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# Vocabulary
VOCAB = {
    '<PAD>': 0, 'go': 1, 'to': 2, 'the': 3,
    'red': 4, 'blue': 5, 'green': 6,
    'circle': 7, 'triangle': 8, 'square': 9,
}
VOCAB_SIZE = len(VOCAB)
MAX_SEQ_LEN = 5

def tokenize_text(text: str, max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
    words = text.lower().split()
    tokens = [VOCAB.get(w, 0) for w in words]
    # Pad
    if len(tokens) < max_len:
        tokens += [VOCAB['<PAD>']] * (max_len - len(tokens))
    return torch.tensor(tokens[:max_len], dtype=torch.long)

def load_image(path: str) -> torch.Tensor:
    try:
        img = Image.open(path).convert('RGB')
        img = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(img.transpose(2, 0, 1)) # (C, H, W)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return torch.zeros(3, 128, 128)

class NeuralNavigatorDataset(Dataset):
    def __init__(self, data_dir: str, normalize_coords: bool = True):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'images')
        self.annotations_dir = os.path.join(data_dir, 'annotations')
        self.normalize_coords = normalize_coords
        
        with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            
        self.samples = []
        print(f"Filtering black images from {self.images_dir}...")
        for s in metadata['samples']:
            img_path = os.path.join(self.images_dir, s['image_file'])
            try:
                # Fast check for black image
                with Image.open(img_path) as img:
                    if img.getextrema() == ((0, 0), (0, 0), (0, 0)): # RGB min/max are all 0
                        continue
                self.samples.append(s)
            except Exception:
                continue
        print(f"Loaded {len(self.samples)} valid samples (filtered {len(metadata['samples']) - len(self.samples)} black/bad images)")
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load Image
        image_path = os.path.join(self.images_dir, sample['image_file'])
        image = load_image(image_path)
        
        # Load Text
        text_tokens = tokenize_text(sample['text'])
        
        # Load Path
        with open(os.path.join(self.annotations_dir, sample['annotation_file']), 'r') as f:
            path = torch.tensor(json.load(f)['path'], dtype=torch.float32)

        if self.normalize_coords:
            # Normalize [-1, 1] (Assuming 128x128 image, center is 64)
            path = (path / 64.0) - 1.0

        return image, text_tokens, path

class NeuralNavigatorTestDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'images')
        with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
            self.samples = json.load(f)['samples']
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = load_image(os.path.join(self.images_dir, sample['image_file']))
        text_tokens = tokenize_text(sample['text'])
        text_command = sample['text']

        return image, text_tokens, text_command, sample['image_file']

def get_test_dataloader(test_dir, batch_size=1, num_workers=0):
    dataset = NeuralNavigatorTestDataset(test_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def get_dataloaders(data_dir, batch_size=32, val_split=0.1, num_workers=0, seed=42):
    dataset = NeuralNavigatorDataset(data_dir)
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    
    train, val = torch.utils.data.random_split(
        dataset, [n_train, n_val], 
        generator=torch.Generator().manual_seed(seed)
    )
    
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )
