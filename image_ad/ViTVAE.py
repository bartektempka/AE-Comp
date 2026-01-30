import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np

class ViTVAE(pl.LightningModule):
    def __init__(self, image_size=256, patch_size=32, input_channels=3, 
                 dim=256, depth=4, heads=4, mlp_dim=512, latent_dim=512, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.latent_dim = latent_dim
        
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = input_channels * patch_size ** 2
        
        # Patch Embedding
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(input_channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2), # (B, dim, num_patches)
        )
        
        # Positional Embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Latent Space
        self.fc_mu = nn.Linear(dim, latent_dim)
        self.fc_logvar = nn.Linear(dim, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, dim * self.num_patches)
        self.pos_embedding_decoder = nn.Parameter(torch.randn(1, self.num_patches, dim))
        
        decoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=depth)
        
        self.to_pixels = nn.Linear(dim, patch_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, img):
        # Encoder
        x = self.to_patch_embedding(img) # (B, dim, num_patches)
        x = x.transpose(1, 2) # (B, num_patches, dim)
        b, n, _ = x.shape
        
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        
        encoded = self.transformer_encoder(x)
        
        # Use CLS token for latent space
        cls_output = encoded[:, 0]
        mu = self.fc_mu(cls_output)
        logvar = self.fc_logvar(cls_output)
        
        z = self.reparameterize(mu, logvar)
        
        # Decoder
        x_rec = self.decoder_input(z).view(b, self.num_patches, self.dim)
        x_rec += self.pos_embedding_decoder
        
        decoded = self.transformer_decoder(x_rec)
        
        # Project back to pixels
        pixels = self.to_pixels(decoded) # (B, num_patches, patch_dim)
        
        return self.patches_to_image(pixels), mu, logvar

    def patches_to_image(self, patches):
        # patches: (B, num_patches, patch_dim)
        b, n, _ = patches.shape
        h = w = self.image_size // self.patch_size
        c = 3
        p = self.patch_size
        
        x = patches.view(b, h, w, c, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5) # (B, C, H, P, W, P)
        x = x.reshape(b, c, h * p, w * p)
        return torch.sigmoid(x)

    def loss_function(self, x_hat, x, mu, logvar):
        recon_loss = F.mse_loss(x_hat, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
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
