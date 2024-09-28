import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embd_dim, num_heads = 8):
        super(MultiHeadSelfAttention, self).__init__()

        assert embd_dim % num_heads == 0, "Embedding Dimension (embd_dim) must be divisible by num_heads"

        self.embd_dim = embd_dim
        self.num_heads = num_heads
        self.hidden_dim = embd_dim // num_heads
        self.scale = self.hidden_dim ** 0.5

        self.W_Q = nn.Linear(in_features=embd_dim, out_features=embd_dim)
        self.W_K = nn.Linear(in_features=embd_dim, out_features=embd_dim)
        self.W_V = nn.Linear(in_features=embd_dim, out_features=embd_dim)

        self.Mat_Mul = nn.Linear(in_features=self.embd_dim, out_features=self.embd_dim)
    
    def forward(self, H, mask = None):

        batch_size, seq_length, embd_dim = H.size() # H is (batch_size, T, h) -> (batch_size, sequence_length, embedding dim)

        Q = self.W_Q(H).view(batch_size, seq_length, self.num_heads, self.hidden_dim).transpose(1, 2)
        K = self.W_K(H).view(batch_size, seq_length, self.num_heads, self.hidden_dim).transpose(1, 2)
        V = self.W_V(H).view(batch_size, seq_length, self.num_heads, self.hidden_dim).transpose(1, 2)
        # Now Q, K, V are (batch_size, num_heads, seq_length, hidden_dim)

        QK_t = torch.matmul(Q, K.transpose(-1, -2))
        if mask is not None:
            dot_product = QK_t.masked_fill(mask==0, -torch.inf) / self.scale
        else:
            dot_product = QK_t / self.scale
        scaled_dot_product = torch.softmax(dot_product, dim=-1)
        head_output = torch.matmul(scaled_dot_product, V).transpose(1, 2)
        concatenated_heads = head_output.contiguous().view(batch_size, seq_length, embd_dim)
        context_aware_representation = self.Mat_Mul(concatenated_heads)
        return context_aware_representation


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embd_dim, num_heads = 8):
        super(MultiHeadCrossAttention, self).__init__()

        assert embd_dim % num_heads == 0, "Embedding Dimension (embd_dim) must be divisible by num_heads"

        self.embd_dim = embd_dim
        self.num_heads = num_heads
        self.hidden_dim = embd_dim // num_heads
        self.scale = self.hidden_dim ** 0.5

        self.W_Q = nn.Linear(in_features=embd_dim, out_features=embd_dim)
        self.W_K = nn.Linear(in_features=embd_dim, out_features=embd_dim)
        self.W_V = nn.Linear(in_features=embd_dim, out_features=embd_dim)

        self.Mat_Mul = nn.Linear(in_features=self.embd_dim, out_features=self.embd_dim)
    
    def forward(self, S, E):
        batch_size, seq_length, embd_dim = S.size() # S/E is (batch_size, T, h) -> (batch_size, sequence_length, embedding dim)
        
        Q = self.W_Q(S).view(batch_size, seq_length, self.num_heads, self.hidden_dim).transpose(1, 2)
        K = self.W_K(E).view(batch_size, seq_length, self.num_heads, self.hidden_dim).transpose(1, 2)
        V = self.W_V(E).view(batch_size, seq_length, self.num_heads, self.hidden_dim).transpose(1, 2)

        QK_t = torch.matmul(Q, K.transpose(-1, -2))
        dot_product = QK_t / self.scale
        scaled_dot_product = torch.softmax(dot_product, dim=-1)
        head_output = torch.matmul(scaled_dot_product, V).transpose(1, 2)
        concatenated_heads = head_output.contiguous().view(batch_size, seq_length, self.embd_dim)
        outputs = self.Mat_Mul(concatenated_heads)
        return outputs