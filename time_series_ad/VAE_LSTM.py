import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl


def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()


class VAE_LSTM(pl.LightningModule):
    def __init__(self, seq_len, n_features, hidden_size, latent_dim, lr=0.001):
        super(VAE_LSTM, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # Latent space layers
        self.fc_mu = nn.Linear(hidden_size * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size * 2, latent_dim)

        # Decoder
        self.fc_latent_to_hidden = nn.Linear(latent_dim, hidden_size)

        self.decoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, n_features)
        self.lr = lr

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + std * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Encode
        _, (h, c) = self.encoder(x)
        # print(f"Encoder hidden state shape: {h.shape}")
        # print(f"Encoder cell state shape: {c.shape}")
        h_last = h[-1]  # Get the last layer's hidden state
        c_last = c[-1]  # Get the last layer's cell state
        combined_hidden = torch.cat((h_last, c_last), dim=1)
        # print(f"Combined hidden state shape: {combined_hidden.shape}")

        # Get latent parameters
        mu = self.fc_mu(combined_hidden)
        logvar = self.fc_logvar(combined_hidden)

        # Sample from latent space
        z = self.reparameterize(mu, logvar)
        # print(f"Latent vector shape: {z.shape}")

        # Decode
        h_dec = self.fc_latent_to_hidden(z).unsqueeze(0)
        c_dec = self.fc_latent_to_hidden(z).unsqueeze(0)

        decoder_input = torch.zeros(batch_size, 1, self.n_features).to(self.device)
        outputs = torch.zeros_like(x).to(self.device)

        hidden_state = (h_dec, c_dec)

        for t in range(seq_len):
            decoder_output, hidden_state = self.decoder(decoder_input, hidden_state)
            decoder_input = self.fc(self.relu(decoder_output))
            outputs[:, t, :] = decoder_input.squeeze(1)

        assert (
            outputs.shape == x.shape
        ), f"Output shape {outputs.shape} does not match input shape {x.shape}"

        # return outputs, mu, logvar
        return outputs, mu, logvar

    def training_step(self, batch, batch_idx):
        (x,) = batch
        x_hat, mu, logvar = self.forward(x)
        loss = nn.MSELoss()(x_hat, x) + kl_divergence(mu, logvar)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (x,) = batch
        x_hat, mu, logvar = self.forward(x)
        loss = nn.MSELoss()(x_hat, x) + kl_divergence(mu, logvar)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
