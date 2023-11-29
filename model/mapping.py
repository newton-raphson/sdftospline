import torch
import torch.nn as nn
import torch.nn.functional as F

class MappingNetwork(nn.Module):
    def __init__(self, latent_dim, hidden_dim, style_dim, num_domains):
        super(MappingNetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, style_dim),
            nn.ReLU()
        )
        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared.append(
                nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, style_dim),
                    nn.ReLU()
                )
            )
    
    def forward(self, x, domain=None):
        if domain is None:
            out = self.shared(x)
        else:
            out = self.unshared[domain](x)
        return out