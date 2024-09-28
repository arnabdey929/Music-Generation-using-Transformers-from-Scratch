import torch
import torch.nn as nn
from PersonalAttention import MultiHeadSelfAttention

class Encoder(nn.Module):
    def __init__(self, seq_length, embd_dim, num_heads = 8, dropout = 0.1):
        super(Encoder, self).__init__()

        assert embd_dim % num_heads == 0, "Embedding Dimension (embd_dim) must be divisible by num_heads"
        
        self.SelfAttention = MultiHeadSelfAttention(embd_dim=embd_dim, num_heads=num_heads)
        self.FeedForward = nn.Sequential(
            nn.Linear(in_features=embd_dim, out_features=4 * embd_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=4*embd_dim, out_features=embd_dim)
        )
        self.LayerNorm = nn.LayerNorm(normalized_shape=(seq_length, embd_dim))
    
    def forward(self, H):
        context_aware_representation = H + self.SelfAttention(H)
        intermediate_output = self.LayerNorm(context_aware_representation)
        before_Add_Norm = intermediate_output + self.FeedForward(intermediate_output)

        finalOutput = self.LayerNorm(before_Add_Norm)
        return finalOutput