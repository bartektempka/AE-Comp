import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

class VAE(pl.LightningModule):
    def __init__(self, input_channels=3, latent_dim=1024, lr=1e-3):
        super(VAE, self).__init__()
        self.lr = lr
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1), # 128x128
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 16x16
            nn.ReLU(),
        )
        
        self.flatten_size = 256 * 16 * 16
        
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1), # 256x256
            nn.Sigmoid()
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x_encoded = self.encoder_conv(x)
        x_flat = x_encoded.view(x_encoded.size(0), -1)
        
        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)
        
        z = self.reparameterize(mu, logvar)
        
        z_decoded = self.fc_decode(z)
        z_reshaped = z_decoded.view(z_decoded.size(0), 256, 16, 16)
        
        x_hat = self.decoder_conv(z_reshaped)
        return x_hat, mu, logvar

    def loss_function(self, x_hat, x, mu, logvar):
        # Reconstruction loss
        recon_loss = F.mse_loss(x_hat, x, reduction='sum')
        
        # KL Divergence
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kld_loss, recon_loss, kld_loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mu, logvar = self(x)
        loss, recon_loss, kld_loss = self.loss_function(x_hat, x, mu, logvar)
        
        # Normalize loss by batch size for logging consistency
        self.log('train_loss', loss / x.size(0))
        self.log('train_recon_loss', recon_loss / x.size(0))
        self.log('train_kld_loss', kld_loss / x.size(0))
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mu, logvar = self(x)
        loss, recon_loss, kld_loss = self.loss_function(x_hat, x, mu, logvar)
        
        self.log('val_loss', loss / x.size(0))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
