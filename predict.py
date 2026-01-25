"""
Neural Navigator Prediction Script
"""

import os
import argparse
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from data_loader import (
    get_test_dataloader,
    load_image,
    tokenize_text
)
from model import NeuralNavigator

def load_model(checkpoint_path, device):
    model = NeuralNavigator()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def predict_path(model, image, text_tokens, device, image_size=128.0):
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        text_tokens = text_tokens.unsqueeze(0).to(device)
        
        pred_path, attn_map, _ = model(image, text_tokens)
        
        # Process Path
        pred_path = pred_path.squeeze(0).cpu().numpy()
        pred_path = (pred_path + 1.0) * (image_size / 2.0)
        
        # Process Attention
        # attn_map: (B, 64) -> (64,)
        attn_map = attn_map.squeeze(0).cpu().numpy()
        
    return pred_path, attn_map

def visualize_attention(image_path, attn_map, save_path):
    """Overlay attention heatmap on image."""
    img = Image.open(image_path).convert('RGB')

    # Reshape (64,) -> (8, 8) and normalize
    attn_grid = np.asarray(attn_map).reshape(8, 8)
    attn_norm = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min() + 1e-8)

    # Upsample to image size
    attn_img = Image.fromarray((attn_norm * 255).astype(np.uint8))
    attn_img = attn_img.resize(img.size, resample=Image.NEAREST)

    # Apply colormap and overlay
    cmap = plt.get_cmap('jet')
    heatmap = (cmap(np.asarray(attn_img) / 255.0)[:, :, :3] * 255).astype(np.uint8)
    overlay = Image.blend(img, Image.fromarray(heatmap), alpha=0.4)
    overlay.save(save_path)

def visualize_path(image_path, pred_path, text_command, save_path):
    img = Image.open(image_path).convert('RGB')
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    
    # Draw path
    ax.plot(pred_path[:, 0], pred_path[:, 1], 'lime', linewidth=2.5, label='Path')
    ax.scatter(pred_path[0, 0], pred_path[0, 1], c='cyan', s=100, marker='o', edgecolors='k', label='Start')
    ax.scatter(pred_path[-1, 0], pred_path[-1, 1], c='yellow', s=120, marker='*', edgecolors='k', label='End')
    ax.scatter(pred_path[1:-1, 0], pred_path[1:-1, 1], c='lime', s=30, marker='o', edgecolors='darkgreen')
    
    ax.set_title(f'"{text_command}"', fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def run_inference(model, test_dir, output_dir, device, num_samples=None):
    os.makedirs(output_dir, exist_ok=True)
    loader = get_test_dataloader(test_dir, batch_size=1)
    
    n_process = num_samples if num_samples else len(loader.dataset)
    print(f"Processing {n_process} samples...")
    
    results = []
    for idx, (image, text_tokens, text_command, image_file) in enumerate(loader):
        if idx >= n_process: break
        
        pred_path, attn_map = predict_path(model, image.squeeze(0), text_tokens.squeeze(0), device)
        
        # Extract metadata
        image_file = image_file[0]
        text_command = text_command[0]
        
        img_full_path = os.path.join(test_dir, 'images', image_file)
        
        # Visualize Path
        visualize_path(img_full_path, pred_path, text_command, os.path.join(output_dir, f'path_{image_file}'))
        
        # Visualize Attention
        visualize_attention(img_full_path, attn_map, os.path.join(output_dir, f'attn_{image_file}'))
        
        results.append({
            'image': image_file, 'text': text_command,
            'path': pred_path.tolist()
        })
        
        if (idx+1) % 10 == 0: print(f"Processing {idx+1}/{n_process}")
        
    with open(os.path.join(output_dir, 'predictions.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved predictions and attention maps -> {output_dir}")

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    model = load_model(args.checkpoint, device)
    
    if args.single_image:
        print(f"Predicting for single image: {args.single_image}")
        img = load_image(args.single_image)
        tok = tokenize_text(args.text_command)
        
        path, attn = predict_path(model, img, tok, device)
        
        visualize_path(args.single_image, path, args.text_command, args.output_path or 'pred_path.png')
        visualize_attention(args.single_image, attn, 'pred_attn.png')
        print(f"Saved: {args.output_path or 'pred_path.png'} and pred_attn.png")
    else:
        run_inference(model, args.test_dir, args.output_dir, device, args.num_samples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pth')
    parser.add_argument('--test_dir', default='dataset_repo/test_data')
    parser.add_argument('--output_dir', default='predictions')
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--single_image')
    parser.add_argument('--text_command', default='Go to the Red Circle')
    parser.add_argument('--output_path')
    args = parser.parse_args()
    main(args)
