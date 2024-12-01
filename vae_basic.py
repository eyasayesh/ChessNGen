import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, negative_slope = 0.01):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.in_norm = nn.InstanceNorm2d(out_channels)  # Independent of batch
        self.leaky_relu = nn.LeakyReLU(negative_slope,inplace=True)

    def forward(self, x):
        return self.leaky_relu(self.in_norm(self.conv(x)))

@dataclass
class VAEConfig:
    input_size: int = 256
    in_channels: int = 3
    latent_dim: int = 256
    hidden_dims: list = None
    max_lr: float = 1e-3
    kld_weight: float = 0.2
    negative_slope: float = 0.1
    weight_decay: float = 1e-6
    adamw_betas: tuple = (0.9, 0.999)
    epochs: int = 10
    minibatch_size: int = 32
    minibatch_num: int = 4
    val_split: float = 0.1
    num_workers: int = 3
    save_epochs: int = 5
    warmup_ratio: float = 0.3  # Spend 30% of training in warmup
    div_factor: int = 25  # initial_lr = max_lr/25
    final_div_factor: int = 1000  # final_lr = initial_lr/1000
    anneal_strategy: chr = 'cos'

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [32, 64, 128, 256]

class ChessVAE(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        
        # Initialize empty lists for encoder and decoder layers
        encoder_layers = []
        decoder_layers = []
        current_channels = config.in_channels

        # Build Encoder
        for h_dim in config.hidden_dims:
            encoder_layers.extend([
                ConvBlock(current_channels, h_dim),
                nn.MaxPool2d(2)
            ])
            current_channels = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Calculate final spatial dimensions
        self.final_spatial_dim = config.input_size // (2 ** len(config.hidden_dims))
        final_dim = config.hidden_dims[-1] * (self.final_spatial_dim ** 2)
        
        # Latent space projections
        self.fc_mu = nn.Linear(final_dim, config.latent_dim)
        self.fc_var = nn.Linear(final_dim, config.latent_dim)
        
        # Initial decoder dense layer
        self.decoder_input = nn.Linear(config.latent_dim, final_dim)
        
        # Build Decoder
        decoder_hidden_dims = config.hidden_dims.copy()
        decoder_hidden_dims.reverse()
        decoder_in_channels = decoder_hidden_dims[0]
        
        for i in range(len(decoder_hidden_dims) - 1):
            decoder_layers.extend([
                nn.Upsample(scale_factor=2),
                ConvBlock(decoder_hidden_dims[i], decoder_hidden_dims[i + 1])
            ])
        
        # Final layers
        decoder_layers.extend([
            nn.Upsample(scale_factor=2),
            nn.Conv2d(decoder_hidden_dims[-1], 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            #Kaiming initialization
            nn.init.kaiming_normal_(
                m.weight,
                mode = 'fan_out',
                nonlinearity='leaky_relu',
                a = self.config.negative_slope
            )
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
        elif isinstance(m, nn.Linear):
            # Xavier initialization for linear layers
            nn.init.xavier_normal_(m.weight,gain = 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            

    #input x should be normalized on a scale of 0-1
    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, self.config.hidden_dims[-1], 
                   self.final_spatial_dim, 
                   self.final_spatial_dim)
        x = self.decoder(x)
        return x
    
    #reparemetrization trick
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def training_step(self, batch    ):
        x = batch
        recon_x, mu, log_var = self(x)
        
        # Reconstruction loss (using MSE for RGB images)
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        
        # KL Divergence loss
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1))
        
        # Total loss
        loss = recon_loss + self.config.kld_weight * kld_loss
       
        return loss

    def validation_step(self, batch):
        x = batch
        recon_x, mu, log_var = self(x)
        
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1))
        loss = recon_loss + self.config.kld_weight * kld_loss
        return loss

    def configure_optimizers(self,train_loader):
        # Setup AdamW optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.max_lr,
            weight_decay=self.config.weight_decay,
            betas=self.config.adamw_betas
        )
        scheduler =  OneCycleLR(
        optimizer,
        max_lr=self.config.max_lr,
        epochs=self.config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=self.config.warmup_ratio,  # Spend 30% of training in warmup
        div_factor=self.config.div_factor,  # initial_lr = max_lr/25
        final_div_factor=self.config.final_div_factor,  # final_lr = initial_lr/1000
        anneal_strategy=self.config.anneal_strategy  # Cosine annealing
    )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

if __name__ == "__main__":
    config = VAEConfig()
    print