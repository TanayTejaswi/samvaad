import torch
import torch.nn as nn

class SignNetGRU(nn.Module):
    def __init__(self, input_dim: int = 254, hidden_dim: int = 128, num_classes: int = 50):
        super().__init__()
        # 2-layer bidirectional GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim // 2, # bidirectional doubles it
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        # Penultimate embedding layer
        self.fc_embed = nn.Linear(hidden_dim, 128)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.3)
        # Final classifier
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (B, T, F)
        out, _ = self.gru(x)
        
        # Max pooling over time (highly NPU friendly, no dynamic indexing)
        out = torch.max(out, dim=1)[0] # shape: (B, hidden_dim)
        
        # 128-d embedding
        emb = self.drop(self.act(self.fc_embed(out)))
        
        # Logits
        logits = self.fc_out(emb)
        
        return logits, emb

class SignNetTransformer(nn.Module):
    def __init__(self, input_dim: int = 254, d_model: int = 128, num_classes: int = 50, seq_len: int = 48):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        
        # Learned positional embedding (seq_len + 1 for CLS token)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len + 1, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True # Better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Final classifier
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (B, T, F)
        B, T, _ = x.shape
        
        x = self.proj(x) # (B, T, d_model)
        
        # Expand CLS token to batch size
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, d_model)
        
        # Concat CLS token
        x = torch.cat((cls_tokens, x), dim=1) # (B, T+1, d_model)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        x = self.layer_norm(x)
        
        # Extract CLS token output (index 0)
        emb = x[:, 0, :] # (B, d_model) 128-d embedding
        
        # Logits
        logits = self.fc_out(emb)
        
        return logits, emb
