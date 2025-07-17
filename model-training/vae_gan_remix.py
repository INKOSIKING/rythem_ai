import torch
import torch.nn as nn

class VAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, output_dim)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        h = torch.relu(self.fc2(h))
        return torch.tanh(self.fc3(h))

class VAEDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return torch.sigmoid(self.fc3(h))

class VAEGANRemixer(nn.Module):
    """
    VAE-GAN for music/audio remixing and creative generation.
    """
    def __init__(self, input_dim, latent_dim=128):
        super().__init__()
        self.encoder = VAEEncoder(input_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, input_dim)
        self.discriminator = VAEDiscriminator(input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        validity = self.discriminator(recon)
        return recon, mu, logvar, validity

    def remix(self, x, remix_strength=0.5):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        z_remix = z + remix_strength * torch.randn_like(z)
        return self.decoder(z_remix)