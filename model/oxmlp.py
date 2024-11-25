# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalEmbedding(nn.Module):
    def __init__(self, size, scale=1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x):
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.Tensor([10000.0])) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0).to(device=x.device)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)

        return emb

    def __len__(self):
        return self.size


class OXMLP(nn.Module):
    """
    MLP neural network implementation with skip connections.
    """
    def __init__(self, 
                 input_dim=2, 
                 hidden_dims=[128], 
                 output_dim=2, 
                 activation='gelu'
            ):
        super(OXMLP, self).__init__()

        activations = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'silu': nn.SiLU,
            'tanh': nn.Tanh,
            'sin': lambda: nn.Sin(),
        }
        assert activation in activations, f"Activation '{activation}' not supported."
        
        # Set activation function
        self.act = activations[activation]()

        # Build layers with skip connections
        self.layers = nn.ModuleList()
        self.skip_connections = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 2):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            self.skip_connections.append(dims[i] == dims[i+1])  # Skip only if dimensions match

        # Final output layer (no skip connection)
        self.final_layer = nn.Linear(dims[-2], dims[-1])

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers):
            residual = out if self.skip_connections[i] else 0  # Add residual if dimensions match
            out = layer(out)
            out = self.act(out)
            out = out + residual

        out = self.final_layer(out)  # Final layer without skip connection
        return out


class CMLP(nn.Module):
    """
        Concat MLP diffusion model.
    """
    def __init__(self,
                 input_dim=2,
                 hidden_dims=[128],
                 output_dim=2,
                 activation="relu",
                 manifold=None
            ):
        
        super(CMLP, self).__init__()

        self.input_dim = input_dim

        self.manifold = manifold

        self.net = OXMLP(input_dim+1, hidden_dims, output_dim, activation)

    def forward(self, x, t):
        """
            x: Tensor of shape [batch_size, input_dim]
            t: Tensor of shape [batch_size]
        """
        # Concatenate x and t embeddings
        t_emb = t*999.0

        h = torch.cat([x, t_emb.view(-1,1)], dim=-1)          # [batch_size, embedding_dim * 2]

        # Pass through the network
        out = self.net(h) 

        # Ensure output is in the tangent space of the manifold
        if self.manifold is not None:
            out = self.manifold.projv(out, x)

        return out


class CEmbedMLP(nn.Module):
    """
        Concat time-embedding MLP diffusion model implementation.
    """
    def __init__(self, 
                 input_dim=2, 
                 embedding_dim=128, 
                 hidden_dims=[128], 
                 output_dim=2, 
                 activation='gelu', 
                 manifold=None,
            ):
        
        super(CEmbedMLP, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        self.manifold = manifold  # The manifold is the disc in our case
      
        self.x_encoder = OXMLP(input_dim, [embedding_dim], embedding_dim, activation)
        self.t_encoder = OXMLP(embedding_dim, [], embedding_dim, activation)
        self.t_emb_fn = SinusoidalEmbedding(embedding_dim)
        self.net = OXMLP(2*embedding_dim, 2*hidden_dims, output_dim, activation)

    def forward(self, x, t):
        """
        x: Tensor of shape [batch_size, input_dim]
        t: Tensor of shape [batch_size]
        """
        # Time embedding
        t_emb = self.t_emb_fn(t)                               # [batch_size, embedding_dim]
        t_encoded = self.t_encoder(t_emb)                      # [batch_size, embedding_dim]

        # x encoding
        x_encoded = self.x_encoder(x)                          # [batch_size, embedding_dim]

        # Concatenate x and t embeddings
        h = torch.cat([x_encoded, t_encoded], dim=-1)          # [batch_size, embedding_dim * 2]

        # Pass through the network
        out = self.net(h) 

        # Ensure output is in the tangent space of the manifold
        if self.manifold is not None:
            out = self.manifold.projv(out, x)

        return out
