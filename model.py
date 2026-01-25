"""
Neural Navigator - Multi-Modal Transformer Model (Fixed)

Key Changes:
1. Cross-Attention Fusion: Text queries attend over image patches
2. Stronger text signal before fusion
3. Direct concatenation of attended features for decoder
"""

import math
import torch
import torch.nn as nn
from data_loader import VOCAB_SIZE, MAX_SEQ_LEN


class VisionEncoder(nn.Module):
    """
    4-layer CNN: (B, 3, 128, 128) -> (B, 64, d_model)
    Outputs 8x8 patch tokens to preserve spatial information.
    """

    def __init__(self, d_model=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),   # 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, d_model, 3, stride=2, padding=1), # 8x8
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )
        
        # Learnable 2D positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)

    def forward(self, x):
        x = self.conv(x)  # (B, d_model, 8, 8)
        x = x.flatten(2).transpose(1, 2)  # (B, 64, d_model)
        x = x + self.pos_embed
        return x


class TextEncoder(nn.Module):
    """
    Embeds text tokens with learnable positional encoding.
    Returns both token embeddings and a pooled summary.
    """

    def __init__(self, vocab_size=VOCAB_SIZE, d_model=256, max_len=MAX_SEQ_LEN):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
        # Simple self-attention to get a summary
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens):
        # Embed tokens
        x = self.embed(tokens)  # (B, T, d)
        x = x + self.pos_embed[:, :x.size(1)]
        
        # Self-attention to refine
        mask = (tokens == 0)  # Padding mask
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm(x + attn_out)
        
        # Pool: mean over non-padded tokens
        lengths = (tokens != 0).sum(dim=1, keepdim=True).float()
        text_summary = (x * (tokens != 0).unsqueeze(-1).float()).sum(dim=1) / lengths
        
        return x, text_summary  # (B, T, d), (B, d)


class CrossAttentionFusion(nn.Module):
    """
    Text-to-Image Cross-Attention: Text summary queries image patches.
    This forces the model to find the relevant image region for the text.
    """

    def __init__(self, d_model=256, num_heads=4):
        super().__init__()
        # Text queries, Image is key/value
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        # FFN to refine
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, text_summary, image_tokens):
        """
        Args:
            text_summary: (B, d_model) - pooled text representation
            image_tokens: (B, 64, d_model) - image patch tokens
        Returns:
            context: (B, d_model) - fused context vector
            attn_weights: (B, 64) - attention over image patches
        """
        # Expand text_summary to (B, 1, d)
        query = text_summary.unsqueeze(1)
        
        # Cross-attention: text queries over image
        attn_out, attn_weights = self.cross_attn(
            query=query,
            key=image_tokens,
            value=image_tokens,
        )
        # attn_out: (B, 1, d), attn_weights: (B, 1, 64)
        
        # Residual + norm
        context = self.norm1(query + attn_out)
        
        # FFN
        context = self.norm2(context + self.ffn(context))
        
        return context.squeeze(1), attn_weights.squeeze(1)


class PathDecoder(nn.Module):
    """
    LSTM-based decoder that outputs sequence of (x, y) coordinates.
    """

    def __init__(self, d_model=256, num_points=10, hidden_size=256):
        super().__init__()
        self.num_points = num_points
        self.hidden_size = hidden_size
        
        # Initialize LSTM hidden state from context
        self.init_h = nn.Linear(d_model, hidden_size)
        self.init_c = nn.Linear(d_model, hidden_size)
        
        # LSTM cell - takes previous position + context as input
        self.lstm = nn.LSTMCell(input_size=2 + d_model, hidden_size=hidden_size)
        
        # Output head
        self.out = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, context):
        batch_size = context.size(0)
        device = context.device
        
        # Initialize hidden state
        h = self.init_h(context)
        c = self.init_c(context)
        
        # Start position (center = 0, 0 in normalized coords)
        pos = torch.zeros(batch_size, 2, device=device)
        
        path = []
        for t in range(self.num_points):
            # Input: previous position + context
            lstm_input = torch.cat([pos, context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            
            # Predict next position
            next_pos = self.out(h)
            path.append(next_pos)
            pos = next_pos
        
        return torch.stack(path, dim=1)  # (B, num_points, 2)


class NeuralNavigator(nn.Module):
    """
    Main Model: Vision + Text -> Cross-Attention Fusion -> Path Decoder
    """

    def __init__(self, num_points=10, d_model=256):
        super().__init__()
        self.d_model = d_model
        
        self.vision_encoder = VisionEncoder(d_model=d_model)
        self.text_encoder = TextEncoder(d_model=d_model)
        self.fusion = CrossAttentionFusion(d_model=d_model)
        self.decoder = PathDecoder(d_model=d_model, num_points=num_points)
        
        # Auxiliary head for direct endpoint prediction
        self.aux_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Tanh(),
        )

    def forward(self, image, text_tokens):
        # Encode
        img_tokens = self.vision_encoder(image)        # (B, 64, d)
        txt_tokens, txt_summary = self.text_encoder(text_tokens)  # (B, T, d), (B, d)
        
        # Fuse: Text attends to Image
        context, attn_map = self.fusion(txt_summary, img_tokens)  # (B, d), (B, 64)
        
        # Auxiliary endpoint prediction
        aux_target = self.aux_head(context)
        
        # Decode path
        path = self.decoder(context)
        
        return path, attn_map, aux_target


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing Neural Navigator Model (Fixed)...")
    model = NeuralNavigator()
    print(f"Parameters: {count_parameters(model):,}")

    # Smoke test
    img = torch.randn(2, 3, 128, 128)
    txt = torch.randint(1, VOCAB_SIZE, (2, MAX_SEQ_LEN))
    with torch.no_grad():
        out, attn, aux = model(img, txt)
    print(f"Path shape: {out.shape}")       # Should be (2, 10, 2)
    print(f"Attention shape: {attn.shape}") # Should be (2, 64)
    print(f"Aux shape: {aux.shape}")        # Should be (2, 2)
    print("Test passed!")
