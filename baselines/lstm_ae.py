import torch
from torch import nn
import torch.nn.functional as F

# LSTM Autoencoder Baseline

class LSTMAutoencoder(nn.Module):

    def __init__(self, n_channels, hidden_dims,  latent_dim, num_layers=2, dropout=0.1):

        super().__init__()

        self.n_channels = n_channels # C

        # LSTM Encoder : (T, C) -> Hidden Dims

        self.lstm_encoder = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_dims,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Linear : Hidden Dims -> Latent Dim

        self.encoded_to_latent = nn.Linear(hidden_dims, latent_dim)

        # Linear : -> Latent Dim -> Hidden Dims

        self.latent_to_decoded = nn.Linear(latent_dim, hidden_dims)

        # LSTM Decoder : Hidden Dims -> Hidden Dims

        self.lstm_decoder = nn.LSTM(
            input_size=hidden_dims,
            hidden_size=hidden_dims,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Output : Hidden Dims -> (T, C)

        self.output = nn.Linear(hidden_dims, n_channels)


    def anomaly_score(self, x):

        # Anomaly Score -> MSE(Input, Reconstructed)

        reconstructed = self.forward(x) # (B, T, C)

        mse = ((x - reconstructed)**2).mean(dim=(1,2)) # (B,)

        return mse

    def forward(self, x):

        # Observation Shape -> (Batch_Size, Time_Steps, Channels)

        B, T, C = x.shape

        # Encoder

        _, (h_n, _) = self.lstm_encoder(x) # (2, B, hidden_dims)

        h = h_n[-1] # (B, hidden_dims)

        z = self.encoded_to_latent(h) # (B, latent_dim)

        # Decoder

        d = self.latent_to_decoded(z) # (B, hidden_dims)

        d = d.unsqueeze(1).repeat(1, T, 1) # (B, T, hidden_dims)

        out,_ = self.lstm_decoder(d) # (B, T, hidden_dims)

        reconstructed = self.output(out) # (B, T, C)

        return reconstructed
    