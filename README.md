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

---

## Challenges & Solutions

### What was the hardest part of this assignment?

The hardest part was **getting the model to actually use the text input**. For hours, the model appeared to be training (loss was decreasing), but it was actually learning to predict the **mean path** (center of the image) regardless of the text command. The validation loss plateaued at ~0.28 and refused to improve no matter how long we trained (we tried 500+ epochs).

This was frustrating because superficially everything looked correct—the architecture followed the assignment spec, the training loop was standard, and there were no runtime errors. The model just wasn't learning the right thing.

### A Specific Bug: Model Ignoring Text (Mean Prediction Collapse)

**The Bug**: After training for 500+ epochs, the model predicted the same endpoint `(0.028, 0.002)` for ALL text commands:
```
'Go to the Red Circle'    -> Endpoint: (0.028, 0.002)
'Go to the Blue Square'   -> Endpoint: (0.028, 0.002)
'Go to the Green Triangle'-> Endpoint: (0.028, 0.002)
```

**How I Identified It**:
1. **Noticed the plateau**: Validation loss stuck at ~0.28 for 400+ epochs despite training loss improving.
2. **Wrote a diagnostic script** (`diagnose.py`) that:
   - Loaded the same image with 3 different text commands
   - Compared the predicted endpoints
   - Calculated variance across predictions
3. **The variance was 0.0000** — definitive proof the model was ignoring text entirely.

**Root Cause Analysis**:
The original architecture used **self-attention on concatenated tokens** ([CLS] + text + image). This allowed the model to take a shortcut: it learned to ignore the text tokens and just predict the average path from the image statistics. Self-attention doesn't *force* cross-modal interaction—the model found it easier to predict the mean.

**The Fix**:
Switched from self-attention to **explicit Cross-Attention** where:
- The **text summary** acts as the Query
- The **image patches** act as Key/Value
- This forces the model to use text to "look up" relevant image regions

```python
# Before (broken): Self-attention on concatenated tokens
tokens = torch.cat([cls, text_tokens, image_tokens], dim=1)
encoded = self.encoder(tokens)  # Model ignores text

# After (fixed): Cross-attention
context, attn_weights = self.cross_attn(
    query=text_summary,    # Text asks "where?"
    key=image_tokens,      # Image provides spatial info
    value=image_tokens
)
```

**Result**: Validation loss dropped from 0.28 to **0.06** immediately after the fix.

### Other Challenges Encountered

1. **Black Images in Dataset**: Some training images were corrupted (all black). Fixed by adding a filter in `data_loader.py` using `Image.getextrema()`.

---

## Results

| Metric | Value |
|--------|-------|
| Best Validation Loss | ~0.06 |
| RMS Pixel Error | **~6.1 pixels** |
| Training Epochs | 104 (early stopped) |
| Image Size | 128×128 |

The model achieves approximately **6.1 pixel RMS error**, meaning on average each predicted point is within 6 pixels of the ground truth on a 128×128 image (~5% of image size).

---

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

---

## File Structure

```
├── model.py          # Model architecture (Cross-Attention Fusion)
├── data_loader.py    # Dataset loading with black image filter
├── train.py          # Training loop with early stopping
├── predict.py        # Inference & visualization
├── requirements.txt  # Dependencies
├── checkpoints/      # Saved model weights
├── outputs/          # Training loss curves
└── test_predictions/ # Sample predictions
```
