import torch
import torch.nn as nn
from PersonalAttention import MultiHeadSelfAttention, MultiHeadCrossAttention

class Decoder(nn.Module):
    def __init__(self, seq_length, embd_dim, num_heads = 8, dropout = 0.1, masking = True):
        super(Decoder, self).__init__()

        assert embd_dim % num_heads == 0, "Embedding Dimension (embd_dim) must be divisible by num_heads"
        self.seq_length = seq_length
        self.masking = masking

        self.MaskedMultiHeadAttention = MultiHeadSelfAttention(embd_dim=embd_dim, num_heads=num_heads)
        self.CrossAttention = MultiHeadCrossAttention(embd_dim=embd_dim, num_heads=num_heads)
        self.LayerNorm = nn.LayerNorm(normalized_shape=(seq_length, embd_dim))
        self.FeedForward = nn.Sequential(
            nn.Linear(in_features=embd_dim, out_features=4*embd_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=4*embd_dim, out_features=embd_dim)            
        )
    
    def forward(self, H, E):

        if self.masking is True:
            mask = torch.tril(torch.ones((self.seq_length, self.seq_length), dtype=torch.float32, device='cuda'))
            after_masked_attention = self.MaskedMultiHeadAttention(H, mask)
        else:
            after_masked_attention = self.MaskedMultiHeadAttention(H, None)
        output = H + after_masked_attention
        S = self.LayerNorm(output)
        
        after_cross_attention = self.CrossAttention(S, E)
        output = S + after_cross_attention
        toward_feed_forward = self.LayerNorm(output)

        final_outputs = self.LayerNorm(toward_feed_forward + self.FeedForward(toward_feed_forward))

        return final_outputs