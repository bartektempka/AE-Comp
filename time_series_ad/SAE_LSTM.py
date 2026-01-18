import torch.nn as nn
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl



class SAE_LSTM(pl.LightningModule):
    def __init__(self, seq_len, n_features, hidden_size, num_layers=1, lr=0.001):
        super(SAE_LSTM, self).__init__()
        self.save_hyperparameters()
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

        self.test_errors = []

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        _, hidden = self.encoder(x)

        decoder_input = torch.zeros(batch_size, seq_len, self.n_features).to(self.device)

        decoder_output, hidden = self.decoder(decoder_input, hidden)
        decoder_output = self.fc(self.relu(decoder_output))

        return decoder_output

    def training_step(self, batch, batch_idx):
        (x, y) = batch
        x_hat = self.forward(x)
        loss = (1 - y).unsqueeze(-1) * nn.MSELoss(reduction="none")(
            x_hat, x
        ) + y.unsqueeze(-1) * 10 * nn.MSELoss(reduction="none")(x_hat, 1 - x)
        loss = loss.mean()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (x, y) = batch
        x_hat = self.forward(x)
        loss = (1 - y).unsqueeze(-1) * nn.MSELoss(reduction="none")(
            x_hat, x
        ) + y.unsqueeze(-1) * 10 * nn.MSELoss(reduction="none")(x_hat, 1 - x)
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, = batch
        x_hat = self(x)
        rec_error = ((x - x_hat) ** 2).mean(dim=2, keepdim=True)
        self.test_errors.append(rec_error.cpu())


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
