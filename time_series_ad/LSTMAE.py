import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl


class LSTMAE(pl.LightningModule):
    def __init__(self, seq_len, n_features, hidden_size, num_layers=1, lr=0.001):
        super(LSTMAE, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_size = hidden_size

        # Encoder
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Decoder
        self.decoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, n_features)
        self.lr = lr

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        _, hidden = self.encoder(x)

        decoder_input = torch.zeros(batch_size, 1, self.n_features).to(self.device)
        outputs = torch.zeros_like(x).to(self.device)

        for t in range(seq_len):
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            decoder_input = self.fc(self.relu(decoder_output))
            outputs[:, t, :] = decoder_input.squeeze(1)

        assert (
            outputs.shape == x.shape
        ), f"Output shape {outputs.shape} does not match input shape {x.shape}"

        return outputs

    def training_step(self, batch, batch_idx):
        (x,) = batch
        x_hat = self.forward(x)
        loss = nn.MSELoss()(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (x,) = batch
        x_hat = self.forward(x)
        loss = nn.MSELoss()(x_hat, x)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
