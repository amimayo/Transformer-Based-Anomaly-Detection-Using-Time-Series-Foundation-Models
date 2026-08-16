import torch
from torch import nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):

    def __init__(self, patch_length, window_size, stride, d_model, n_channels):

        super().__init__()

        self.patch_length = patch_length
        self.stride = stride
        self.d_model = d_model

        self.n_patches = ((window_size - patch_length) // stride) + 1
        
        self.layer = nn.Linear(patch_length, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_channels, 128, d_model))

    
    def forward(self, x):

        # (B, C, T)

        B, C, T = x.shape

        # (B, C, n_patches, patch_length)

        patches = x.unfold(dimension=-1, size=self.patch_length, step=self.stride)

        n_patches = patches.shape[2]

        patches = patches.reshape(B*C, n_patches, self.patch_length) 

        # (B*C, n_patches, d_model)

        tokens = self.layer(patches)

        tokens = tokens.reshape(B, C, n_patches, self.d_model)

        tokens = tokens + self.pos_embed[: ,: ,:n_patches, :]

        tokens = tokens.reshape(B*C, n_patches, self.d_model)

        return tokens

class PatchTST(nn.Module):

    def __init__(self, n_channels, T, patch_length, stride, d_model, d_ff, n_heads, n_layers):

        super().__init__()

        self.patch_embedding = PatchEmbedding(patch_length, T, stride, d_model, n_channels)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)

        n_patches = ((T - patch_length) // stride) + 1

        self.head = nn.Linear(n_patches*d_model, patch_length*n_patches)

    def anomaly_score(self, x):

        # Anomaly Score -> MSE(Input, Reconstructed)

        reconstructed = self.forward(x) # (B, T, C)

        mse = ((x - reconstructed)**2).mean(dim=(1,2)) # (B,)

        return mse

    def forward(self, x):

        B, C, T = x.shape

        tokens = self.patch_embedding(x) # (B*C, n_patches, d_model)

        encoded = self.transformer(tokens) # (B*C, n_patches, d_model)

        encoded_flat = encoded.reshape(B*C, -1) # (B*C, n_patches*d_model)

        reconstructed_flat = self.head(encoded_flat) # (B*C, patch_length*n_channels)

        reconstructed = reconstructed_flat.reshape(B, C, -1)[..., :T] 

        return reconstructed