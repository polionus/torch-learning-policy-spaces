import torch
import torch.nn as nn
import math

# Source - https://stackoverflow.com/a/77445896
# Posted by Yakov Dan, modified by community. See post 'Timeline' for change history
# Retrieved 2025-12-02, License - CC BY-SA 4.0

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

    

class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, max_seq_len):
        super().__init__()
        
        # 1. The Positional Encoder
        self.pos_encoder = PositionalEncoding(d_model=input_dim, max_len=max_seq_len)
        
        # 2. The Transformer Layer
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=num_heads, 
            batch_first=True
        )

    def forward(self, x):
        # x comes in as raw embeddings/vectors
        
        # ADD POSITIONS HERE
        x = self.pos_encoder(x)
        
        # Then pass to transformer
        output = self.transformer_layer(x)
        return output

# TODO: comment below is from original LEAPS: check if necessary
# replace unmask_idx with unmask_idx2 after verifying identity
def unmask_idx(output_mask_all, first_end_token_idx, max_program_len):
    for p_idx in range(first_end_token_idx.shape[0]):
        t_idx = int(first_end_token_idx[p_idx].detach().cpu().numpy())
        if t_idx < max_program_len:
            output_mask_all[p_idx, t_idx] = True
    return output_mask_all.to(torch.bool)


def unmask_idx2(x):
    seq, seq_len = x
    if seq_len < seq.shape[0]:
        seq[seq_len] = True
        return True
    return False


def init_gru(module: torch.nn.GRU):
    for name, param in module.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param)

def init_lstm(module: torch.nn.LSTM):
    for name, param in module.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param)



def init(module, weight_init, bias_init, gain=1.):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def masked_mean(x, mask, dim=-1, keepdim=False):
    assert x.shape == mask.shape
    return torch.sum(x * mask.float(), dim=dim, keepdim=keepdim) / torch.sum(mask, dim=dim, keepdim=keepdim)


def masked_sum(x, mask, dim=-1, keepdim=False):
    assert x.shape == mask.shape
    return torch.sum(x * mask.float(), dim=dim, keepdim=keepdim)


def add_record(key, value, global_logs):
    if 'logs' not in global_logs['info']:
        global_logs['info']['logs'] = {}
    logs = global_logs['info']['logs']
    split_path = key.split('.')
    current = logs
    for p in split_path[:-1]:
        if p not in current:
            current[p] = {}
        current = current[p]

    final_key = split_path[-1]
    if final_key not in current:
        current[final_key] = []
    entries = current[final_key]
    entries.append(value)


def log_record_dict(usage, log_dict, global_logs):
    for log_key, value in log_dict.items():
        add_record('{}.{}'.format(usage, log_key), value, global_logs)


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)