import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, img_dim, noise_dim):
        super(Generator, self).__init__()
        input_size = noise_dim
        output_size = img_dim
        self.label_embed = nn.Embedding(10, 10)
        self.net = nn.Sequential(
            nn.Linear(input_size + 10, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_size),
            nn.Tanh()
        )

    def forward(self, noise, labels, noise_dim):
        noise = noise.view(noise.size(0), noise_dim)
        condt = self.label_embed(labels)
        x = torch.cat([noise, condt], 1)
        x = self.net(x)
        return x.view(x.size(0), 28, 28)
        

class Discriminator(nn.Module):
    def __init__(self, img_dim, output_dim):
        super(Discriminator, self).__init__()
        input_size = img_dim
        output_size = output_dim
        self.label_embed = nn.Embedding(10, 10)

        self.net = nn.Sequential(
            nn.Linear(input_size + 10, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
            nn.Sigmoid()
        )
        
    def forward(self, images, labels):
        images = images.view(images.size(0), 784)
        condt = self.label_embed(labels)
        x = torch.cat([images, condt], 1)
        x = self.net(x)
        x = x.squeeze()
        return x