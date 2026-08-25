import os
import torch
from torch import nn
import torch.nn.functional as F
from transformers import RobertaModel
from transformers import RobertaConfig
from torch.utils.checkpoint import checkpoint



class CSUEI(nn.Module):
    def __init__(self,
                 roberta_model_path,
                 vocab_size,
                 max_position_embeddings,
                 num_attention_heads,
                 num_hidden_layers,
                 type_vocab_size,
                 smi_dropout,
                 smile_use_esa,
                 smiles_use_gradient_checkpoint,
                 feature_dim,
                 ms_layer,
                 ms_n_embed,
                 ms_n_heads,
                 ms_dropout,
                 ms_use_se,
                 ms_use_esa,
                 ms_use_gradient_checkpoint,
                 ):
        super(CSUEI, self).__init__()

        self.Smiles_model = SmilesModel(roberta_model_path,vocab_size,max_position_embeddings,num_attention_heads,
                                        num_hidden_layers,type_vocab_size,smi_dropout,smile_use_esa,smiles_use_gradient_checkpoint,feature_dim)

        self.MS_model = MSModel(ms_layer, ms_n_embed, ms_n_heads, ms_dropout, ms_use_se, ms_use_esa, ms_use_gradient_checkpoint)

    def smiles_encoder(self, xis):
        smi_feature, smi_score = self.Smiles_model(xis)
        return smi_feature, smi_score

    def ms_encoder(self, xls):
        ms_feature, ms_score = self.MS_model(xls)
        return ms_feature, ms_score

    def forward(self, smi, ms):
        smi_feature, smi_score = self.smiles_encoder(smi)
        ms_feature, ms_score = self.ms_encoder(ms)

        return smi_feature, smi_score, ms_feature, ms_score


class SmilesModel(nn.Module):
    def __init__(self,
                 roberta_model_path,
                 vocab_size,
                 max_position_embeddings,
                 num_attention_heads,
                 num_hidden_layers,
                 type_vocab_size,
                 smi_dropout,
                 smile_use_esa,
                 smiles_use_gradient_checkpoint,
                 feature_dim,
                 **kwargs
                 ):
        super(SmilesModel, self).__init__(**kwargs)

        self.smiles_config = RobertaConfig(
                vocab_size=vocab_size,
                max_position_embeddings=max_position_embeddings,
                num_attention_heads=num_attention_heads,
                num_hidden_layers=num_hidden_layers,
                type_vocab_size=type_vocab_size,
                hidden_dropout_prob=smi_dropout,
                attention_probs_dropout_prob=smi_dropout,
                hidden_size=feature_dim
            )
        
        self.smiles_model = RobertaModel(config=self.smiles_config, add_pooling_layer=False)

        if roberta_model_path is not None:
            pretrain_roberta_model_dict = torch.load(roberta_model_path)
            self.smiles_model.load_state_dict(pretrain_roberta_model_dict, strict=False)
            print('load pretrain smiles model success')
            
        self.smile_use_esa = smile_use_esa
        self.use_gradient_checkpoint = smiles_use_gradient_checkpoint

        if self.smile_use_esa:
            self.esa = ESA_SMILES(feature_dim, feature_dim)
        else:
            self.dense = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, feature_dim)
            )

    def forward(self, input):
        # ----------------------- gradient checkpoint -----------------------
        if self.training:
            if self.use_gradient_checkpoint:
                self.smiles_model.gradient_checkpointing_enable()
        # print(self.smiles_model.is_gradient_checkpointing)

        input_ids, attention_mask = input  
        input_ids = input_ids.squeeze(1)  
        attention_mask = attention_mask.squeeze(1)

        if self.smile_use_esa:
            hidden_states = self.smiles_model(input_ids, attention_mask)[0]
            smi_feature, smi_score = self.esa(hidden_states, attention_mask)
        else:
            cls_embedding = self.smiles_model(input_ids, attention_mask)[0][:, 0, :]  # cls token
            smi_feature = self.dense(cls_embedding)
            smi_score = None
        return smi_feature, smi_score

    def freeze_backbone(self):
        for param in self.smiles_model.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.smiles_model.parameters():  # open up all params first, then adjust the base parameters
            param.requires_grad = True


#----------------------------------- esa -----------------------------
class ESA_SMILES(nn.Module):
    def __init__(self, feature_dim, out_dim):
        super().__init__()
        self.ln_f = nn.LayerNorm(feature_dim)
        self.linear = nn.Linear(feature_dim, out_dim)
        self.linear1 = nn.Linear(out_dim, out_dim)
        # self.linear2 = nn.Linear(out_dim, out_dim)
        
    def forward(self, hidden_states, attention_mask):
        logits = self.ln_f(hidden_states)  # (B, N, C)
        cap_embes = self.linear(logits) # Q
        features_in = self.linear1(cap_embes) # M
        mask = attention_mask.unsqueeze(-1)  # (B, N, 1)
        features_in = features_in.masked_fill(mask == 0, -1e4)  # (B, N, C)
        features_k_softmax = nn.Softmax(dim=1)(features_in)
        attn = features_k_softmax.masked_fill(mask == 0, 0)
        smi_score = attn * cap_embes
        smi_feature = torch.sum(smi_score, dim=1) # (B, C)

        return smi_feature, smi_score


class ESA_MS(nn.Module):
    def __init__(self, feature_dim, out_dim):
        super().__init__()
        self.ln_f = nn.LayerNorm(feature_dim)
        self.linear = nn.Linear(feature_dim, out_dim)
        self.linear1 = nn.Linear(out_dim, out_dim)
        # self.linear2 = nn.Linear(out_dim, out_dim)

    def forward(self, hidden_states, attention_mask):
        logits = self.ln_f(hidden_states)  # (B, N, C)
        cap_embes = self.linear(logits) # Q
        features_in = self.linear1(cap_embes) # M
        mask = attention_mask.unsqueeze(-1)  # (B, N, 1)
        features_in = features_in.masked_fill(mask == 0, -1e4)  # (B, N, C)
        features_k_softmax = nn.Softmax(dim=1)(features_in)
        attn = features_k_softmax.masked_fill(mask == 0, 0)
        # for cross attention
        ms_score = attn * cap_embes
        ms_feature = torch.sum(attn * cap_embes, dim=1) # (B, C)

        return ms_feature, ms_score


# ---------------transformer ms encoder------------------------
class Attention(nn.Module):
    def __init__(self, n_embed, n_heads, dropout):
        super().__init__()
        head_size = n_embed // n_heads
        self.key = nn.Linear(n_embed, head_size)
        self.query = nn.Linear(n_embed, head_size)
        self.value = nn.Linear(n_embed, head_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ms, ms_mask):
        x = ms  # ms_mask:  B, 100
        # print(x)
        B, T, C = x.shape
        q = self.query(x)  # (B, T, C)
        k = self.key(x)
        wei = q @ k.transpose(-1, -2) / (C ** 0.5)  # (B, T, C) @ (B, C, T) = (B, T, T)
        mask = ms_mask.unsqueeze(1)
        wei = wei.masked_fill(mask == 0, -1e4)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v  # (B, T, T) @ (B, T, C) = (B, T, C)

        return out 


class MultiHead(nn.Module):
    def __init__(self, n_embed, n_heads, dropout):
        super().__init__()
        head_size = n_embed // n_heads
        self.heads = nn.ModuleList([Attention(n_embed, n_heads, dropout) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ms, ms_mask):
        out = torch.cat([h(ms, ms_mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))  # (B, T, C)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_heads, dropout):
        super().__init__()
        # head_size = n_embed // n_heads
        self.sa = MultiHead(n_embed, n_heads, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln = nn.LayerNorm(n_embed)

    def forward(self, ms, ms_mask):
        # ms, ms_mask = pe_output[0], pe_output[1]
        x = self.ln(ms + self.sa(ms, ms_mask))
        x = self.ln1(x + self.ffwd(x)) # residual connection
        return x, ms_mask


class MSEmbedder(nn.Module):
    def __init__(self, max_mz, n_embed, dropout):
        super(MSEmbedder, self).__init__()
        self.mz_embed = nn.Embedding(max_mz, n_embed, padding_idx=0)  

        self.ff = nn.Sequential(nn.Linear(n_embed + 1, n_embed),
                                nn.ReLU(inplace=True),                           
                                nn.Linear(n_embed, n_embed),
                                nn.Dropout(dropout))
        self.d_model = n_embed

    def forward(self, input):
        mz, intensity, ms_mask = input[0].squeeze(1), input[1].squeeze(1).unsqueeze(-1), input[2].squeeze(1)  # mz (B, 100) i (B, 100) ms_mask (B, 100)
        mz_embed = self.mz_embed(mz.long()) # (B, 100, 768)
        x = torch.cat([intensity, mz_embed], dim=2)  # tensor(B, 100,768) ---> (B, 100, 769)
        output = self.ff(x)
        return output, ms_mask  # output (1, 101, 768)


# sinusoidal embedder
class SinusoidalEmbedder(nn.Module):
    def __init__(self, n_embed, dropout):
        super(SinusoidalEmbedder, self).__init__()
        # [tensor(B, 2, 100), tensor(B, 100)]
        # tensor(B, 2, 100) ------> mz:(B, 100), i:(B, 100))
        self.mz_embed = nn.Sequential(nn.Linear(n_embed, n_embed),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(n_embed, n_embed))
        self.ff = nn.Sequential(nn.Linear(n_embed + 1, n_embed),
                                nn.ReLU(inplace=True),                           
                                nn.Linear(n_embed, n_embed),
                                nn.Dropout(dropout))
        self.d_model = n_embed
    def forward(self, input):
        mz, intensity, ms_mask = input[0].squeeze(1), input[1].squeeze(1).unsqueeze(-1), input[2].squeeze(1)  # mz (B, 100) i (B, 100) ms_mask (B, 100)
        B, N = ms_mask.shape
        device = ms_mask.device
        result = torch.ones((B, N, self.d_model), device=device)
        pos = mz.unsqueeze(2)  # mz(B, 100, 1)
        i = torch.arange(0, self.d_model, 2, device=device)
        div = 10000 ** (i / self.d_model)
        term = pos / div  # (B, 300, 384)
        result[:, :, 0::2] = torch.sin(term)
        result[:, :, 1::2] = torch.cos(term)
        # mz_mask = ms_mask.unsqueeze(2)
        # result = result.masked_fill(mz_mask == 0, 0)
        mz_embed = self.mz_embed(result)
        x = torch.cat([intensity, mz_embed], dim=2)  # tensor(B, 100,768) ---> (B, 100, 769)
        output = self.ff(x)
        return output, ms_mask  # output (1, 101, 768)


class MSProjection(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dense = nn.Sequential(
                                    nn.Linear(n_embed, n_embed),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(n_embed, n_embed),
                                   )

    def forward(self, x):
        return self.dense(self.avgpool(x.transpose(1, 2)).squeeze(-1))


class MSModel(nn.Module):
    def __init__(self, n_layer, n_embed, n_heads, dropout, ms_use_se, ms_use_esa, ms_use_gradient_checkpoint):
        super().__init__()
        self.ms_use_se = ms_use_se
        self.ms_use_esa = ms_use_esa
        self.use_gradient_checkpoint = ms_use_gradient_checkpoint
        if self.ms_use_se:
            self.pos_emb = SinusoidalEmbedder(n_embed, dropout)
        else:
            self.pos_emb = MSEmbedder(1660, n_embed, dropout) 
        self.blocks = nn.ModuleList([Block(n_embed, n_heads, dropout) for _ in range(n_layer)])
        if self.ms_use_esa:
            self.esa = ESA_MS(n_embed, n_embed)
        else:
            self.projection = MSProjection(n_embed)

    def forward(self, idx):
        ms, ms_mask = self.pos_emb(idx)  
        if self.use_gradient_checkpoint:
            device = ms.device
            dummy_tensor = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True)
            ms = ms + dummy_tensor
            for block in self.blocks:
                ms, ms_mask = checkpoint(block, ms, ms_mask, use_reentrant=True)
        else:
            for block in self.blocks:
                ms, ms_mask = block(ms, ms_mask)

        if self.ms_use_esa:
            ms_feature, ms_score = self.esa(ms, ms_mask)
        else:
            ms_feature = self.projection(ms)
            ms_score = None
        return ms_feature, ms_score

