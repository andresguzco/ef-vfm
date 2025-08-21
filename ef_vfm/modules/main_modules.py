from typing import Callable, Union

from ef_vfm.modules.transformer import Reconstructor, Tokenizer, Transformer
import torch
import torch.nn as nn
import torch.optim

ModuleType = Union[str, Callable[..., nn.Module]]

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class MLP(nn.Module):
    def __init__(self, d_in, dim_t = 512, use_mlp=True):
        super().__init__()
        self.dim_t = dim_t

        self.proj = nn.Linear(d_in, dim_t)

        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        ) if use_mlp else nn.Linear(dim_t, d_in)

        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
        
        self.use_mlp = use_mlp
    
    def forward(self, x, timesteps):
        emb = self.map_noise(timesteps)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        emb = self.time_embed(emb)
    
        x = self.proj(x) + emb
        return self.mlp(x)
    

class UniModMLP(nn.Module):
    """
        Input:
            x_num: [bs, d_numerical]
            x_cat: [bs, len(categories)]
        Output:
            x_num_pred: [bs, d_numerical], the predicted mean for numerical data
            x_cat_pred: [bs, sum(categories)], the predicted UNORMALIZED logits for categorical data
    """
    def __init__(
            self, d_numerical, categories, num_layers, d_token,
            n_head = 1, factor = 4, bias = True, dim_t=512, use_mlp=True, **kwargs
        ):
        super().__init__()
        self.d_numerical = d_numerical
        self.categories = categories

        self.tokenizer = Tokenizer(d_numerical, categories, d_token, bias = bias)
        self.encoder = Transformer(num_layers, d_token, n_head, d_token, factor)
        d_in = d_token * (d_numerical + len(categories))
        self.mlp = MLP(d_in, dim_t=dim_t, use_mlp=use_mlp)
        self.decoder = Transformer(num_layers, d_token, n_head, d_token, factor)
        self.detokenizer = Reconstructor(d_numerical, categories, d_token)
        
        self.model = nn.ModuleList([self.tokenizer, self.encoder, self.mlp, self.decoder, self.detokenizer])

    def forward(self, x_num, x_cat, timesteps):
        e = self.tokenizer(x_num, x_cat)
        decoder_input = e[:, 1:, :]        # ignore the first CLS token. 
        y = self.encoder(decoder_input)
        pred_y = self.mlp(y.reshape(y.shape[0], -1), timesteps)
        pred_e = self.decoder(pred_y.reshape(*y.shape))
        x_num_pred, x_cat_pred = self.detokenizer(pred_e)
        x_cat_pred = torch.cat(x_cat_pred, dim=-1) if len(x_cat_pred)>0 else torch.zeros_like(x_cat).to(x_num_pred.dtype)

        return x_num_pred, x_cat_pred
