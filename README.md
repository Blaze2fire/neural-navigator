# Neural Navigator

A Multi-Modal AI system that navigates to a specified target object in an image based on natural language commands (e.g., "Go to the Red Circle").

## Model Architecture

The model uses a multi-modal Transformer with Cross-Attention fusion.

```
┌─────────────────┐     ┌─────────────────┐
│  Vision Encoder │     │  Text Encoder   │
│  (4-layer CNN)  │     │  (Embeddings +  │
│  → 64 patches   │     │   Self-Attn)    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         │    ┌──────────────────┘
         │    │ Cross-Attention
         ▼    ▼ (Text queries Image)
    ┌─────────────────┐
    │ Fusion Context  │
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │  Path Decoder   │ → 10 (x,y) coordinates
    │  (LSTM)         │
    └─────────────────┘
```

1. **Vision Encoder**: 4-layer CNN that outputs 64 patch tokens (8×8 grid), preserving spatial information.
2. **Text Encoder**: Token embeddings with positional encoding + self-attention, producing a text summary vector.
3. **Cross-Attention Fusion**: Text summary **queries** image patches to find the target location.
4. **Path Decoder**: LSTM that unrolls 10 (x, y) coordinates, conditioned on the fused context.
5. **Auxiliary Head**: Predicts the endpoint directly to stabilize training.

## Challenges & Solutions

### 1. Data Quality (Black Images)
- **Problem**: Training loss was unstable.
- **Diagnosis**: Found corrupted all-black images in the dataset.
- **Fix**: Added filter in `data_loader.py` using `Image.getextrema()` to skip black images.

### 2. Model Ignoring Text (Mean Prediction Collapse)
- **Problem**: Model predicted the same path (center of image) regardless of text input.
- **Diagnosis**: Ran diagnostic script comparing predictions for different text commands on the same image—variance was zero.
- **Root Cause**: Self-attention on concatenated [CLS, text, image] tokens failed to force text-image interaction.
- **Fix**: Switched to **Cross-Attention** where text directly queries image patches.

### 3. Attention Visualization
- **Purpose**: Verify the model "looks" at the correct object.
- **Implementation**: Extract attention weights from fusion layer, reshape to 8×8, overlay on image.

## Results

| Metric | Value |
|--------|-------|
| Best Validation Loss | ~0.033 |
| RMS Pixel Error | ~3.2 pixels |
| Epochs to Converge | ~150-200 |

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
python train.py --epochs 300 --batch_size 32 --augmentation
```

### Inference
```bash
python predict.py --checkpoint checkpoints/best_model.pth --output_dir test_predictions
```

### Output
- `path_*.png`: Predicted trajectory visualization
- `attn_*.png`: Attention heatmap showing where the model focused
- `predictions.json`: Predicted coordinates

## File Structure

```
├── model.py          # Model architecture
├── data_loader.py    # Dataset loading & preprocessing
├── train.py          # Training loop with early stopping
├── predict.py        # Inference & visualization
├── requirements.txt  # Dependencies
├── checkpoints/      # Saved model weights
├── outputs/          # Training loss curves
└── test_predictions/ # Sample predictions
```
