import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

class VAE_GRF(pl.LightningModule):
    def __init__(self, input_channels=3, latent_channels=16, lr=1e-3):
        super(VAE_GRF, self).__init__()
        self.lr = lr
        self.latent_channels = latent_channels
        
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
        
        # Instead of flattening, we map to latent channels
        self.conv_mu = nn.Conv2d(256, latent_channels, kernel_size=3, padding=1)
        self.conv_logvar = nn.Conv2d(256, latent_channels, kernel_size=3, padding=1)
        
        # Decoder
        self.conv_decode = nn.Conv2d(latent_channels, 256, kernel_size=3, padding=1)
        
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
        
        mu = self.conv_mu(x_encoded)
        logvar = self.conv_logvar(x_encoded)
        
        z = self.reparameterize(mu, logvar)
        
        z_decoded = self.conv_decode(z)
        
        x_hat = self.decoder_conv(z_decoded)
        return x_hat, mu, logvar

    def grf_loss(self, mu, logvar):
        # Gaussian Random Field Prior Loss (Smoothness)
        # We assume a simple 4-neighbor Laplacian precision matrix for the prior
        # Loss ~ mu^T * L * mu + tr(L * Sigma) - log|Sigma|
        
        # Calculate differences between neighbors
        # Horizontal differences
        diff_h = mu[:, :, :, :-1] - mu[:, :, :, 1:]
        # Vertical differences
        diff_v = mu[:, :, :-1, :] - mu[:, :, 1:, :]
        
        smoothness_loss = torch.sum(diff_h ** 2) + torch.sum(diff_v ** 2)
        
        # Variance term (trace of precision * covariance)
        # For standard normal prior, this is sum(sigma^2). 
        # For GRF, it involves neighbors. But if we assume independence in posterior, 
        # and only smoothness in prior, it's complicated.
        # A simplified GRF loss often just adds the smoothness term to the standard KL.
        
        # Standard KL (assuming N(0, I) prior for the non-spatial part, but we want GRF)
        # If we strictly follow GRF prior:
        # KL = 0.5 * ( tr(L*Sigma) + mu^T*L*mu - log|Sigma| + const )
        
        # mu^T * L * mu is exactly the smoothness_loss (sum of squared differences)
        
        # tr(L * Sigma):
        # L has 4 on diagonal, -1 on neighbors.
        # tr(L * Sigma) = 4 * sum(sigma^2) - 2 * sum(covariance between neighbors)
        # Since our posterior is diagonal (independent), covariance between neighbors is 0.
        # So tr(L * Sigma) = 4 * sum(sigma^2) = 4 * sum(exp(logvar))
        
        # log|Sigma| = sum(logvar)
        
        # So GRF KL = 0.5 * ( 4 * sum(exp(logvar)) + smoothness_loss - sum(logvar) )
        
        # However, usually we want to balance this.
        # Let's use a weight for the smoothness.
        
        sigma2 = torch.exp(logvar)
        kl_loss = 0.5 * (4 * torch.sum(sigma2) + smoothness_loss - torch.sum(logvar))
        
        return kl_loss

    def loss_function(self, x_hat, x, mu, logvar):
        # Reconstruction loss
        recon_loss = F.mse_loss(x_hat, x, reduction='sum')
        
        # GRF KL Divergence
        kld_loss = self.grf_loss(mu, logvar)
        
        return recon_loss + kld_loss, recon_loss, kld_loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mu, logvar = self(x)
        loss, recon_loss, kld_loss = self.loss_function(x_hat, x, mu, logvar)
        
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
