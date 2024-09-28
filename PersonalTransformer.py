import torch
import torch.nn as nn
from PersonalEncoder import Encoder
from PersonalDecoder import Decoder

class Transformer(nn.Module):
    def __init__(self, embd_dim, seq_length, vocab_size : int, applySoftmax = False, num_heads_encoder = 8, num_heads_decoder = 8):
        super(Transformer, self).__init__()
        
        self.embd_dim = embd_dim
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.applySoftmax = applySoftmax

        self.encoder = Encoder(seq_length=seq_length, embd_dim=embd_dim, num_heads=num_heads_encoder)
        self.decoder = Decoder(seq_length=seq_length, embd_dim=embd_dim, num_heads=num_heads_decoder)
        self.linear = nn.Linear(in_features=embd_dim, out_features=vocab_size)
    
    def sinusoidal_embedding(self, x):

        _, T_dim, d_model = x.size()
        device = x.device
        log_ten_thousand = torch.log(torch.tensor(10000.0, dtype=torch.float32, device=device))

        pos = torch.arange(T_dim, dtype=torch.float32, device=device).unsqueeze(dim=1)
        denominator = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device) * -(log_ten_thousand / d_model))
        PE = torch.zeros((T_dim, d_model), dtype=torch.float32, device=device)
        PE[:, 0::2] = torch.sin(pos * denominator)
        PE[:, 1::2] = torch.cos(pos * denominator)
        PE = PE.unsqueeze(dim=0)

        return x + PE
    
    def forward(self, inputs, outputs_shifted_right):

        inputs = self.sinusoidal_embedding(inputs)
        outputs_shifted_right = self.sinusoidal_embedding(outputs_shifted_right)

        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded, outputs_shifted_right)
        output_logits = self.linear(decoded)
        if self.applySoftmax is True:
            return torch.softmax(output_logits, dim=-1)
        else:
            return output_logits