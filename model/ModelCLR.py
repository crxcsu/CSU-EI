import torch
from torch import nn
import torch.nn.functional as F
from transformers import RobertaModel
from transformers import RobertaConfig
from torch.utils.checkpoint import checkpoint


class ESA_SMILES(nn.Module):
    def __init__(self, feature_dim, out_dim):
        super().__init__()
        self.ln_f = nn.LayerNorm(feature_dim)
        self.linear = nn.Linear(feature_dim, out_dim)
        self.linear1 = nn.Linear(out_dim, out_dim)

    def forward(self, hidden_states, attention_mask):
        logits = self.ln_f(hidden_states)  # (B, N, C)
        cap_embes = self.linear(logits)  # Q
        features_in = self.linear1(cap_embes)  # M
        mask = attention_mask.unsqueeze(-1)  # (B, N, 1)
        features_in = features_in.masked_fill(mask == 0, -1e4)  # (B, N, C)
        features_k_softmax = nn.Softmax(dim=1)(features_in)
        attn = features_k_softmax.masked_fill(mask == 0, 0)
        smi_score = attn * cap_embes
        smi_feature = torch.sum(smi_score, dim=1)  # (B, C)
        return smi_feature, smi_score


class ESA_MS(nn.Module):
    def __init__(self, feature_dim, out_dim):
        super().__init__()
        self.ln_f = nn.LayerNorm(feature_dim)
        self.linear = nn.Linear(feature_dim, out_dim)
        self.linear1 = nn.Linear(out_dim, out_dim)

    def forward(self, hidden_states, attention_mask):
        logits = self.ln_f(hidden_states)  # (B, N, C)
        cap_embes = self.linear(logits)  # Q
        features_in = self.linear1(cap_embes)  # M
        mask = attention_mask.unsqueeze(-1)  # (B, N, 1)
        features_in = features_in.masked_fill(mask == 0, -1e4)  # (B, N, C)
        features_k_softmax = nn.Softmax(dim=1)(features_in)
        attn = features_k_softmax.masked_fill(mask == 0, 0)
        ms_score = attn * cap_embes
        ms_feature = torch.sum(attn * cap_embes, dim=1)  # (B, C)
        return ms_feature, ms_score


class SmilesRobertaBackbone(nn.Module):
    """
    RoBERTa backbone
    forward(input) -> (seq_hidden:[B,L,C], attention_mask:[B,L])
    """
    def __init__(self,
                 roberta_model_path,
                 vocab_size,
                 max_position_embeddings,
                 num_attention_heads,
                 num_hidden_layers,
                 type_vocab_size,
                 smi_dropout,
                 feature_dim,
                 smiles_use_gradient_checkpoint
                 ):
        super().__init__()
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
            pretrain_roberta_model_dict = torch.load(roberta_model_path, map_location="cpu")
            self.smiles_model.load_state_dict(pretrain_roberta_model_dict, strict=False)
            print('load pretrain smiles model success')

        self.use_gradient_checkpoint = smiles_use_gradient_checkpoint

    def forward(self, input):
        input_ids, attention_mask = input
        input_ids = input_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
        if self.training:
            if self.use_gradient_checkpoint:
                self.smiles_model.gradient_checkpointing_enable()
        hidden_states = self.smiles_model(input_ids, attention_mask)[0]
        return hidden_states, attention_mask

    def freeze_backbone(self):
        for param in self.smiles_model.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.smiles_model.parameters():
            param.requires_grad = True


class SmilesGRUBackbone(nn.Module):
    """
    GRU‑based SMILES backbone, outputs sequence representation for outer aggregation
    forward(input) -> (seq_hidden:[B,L,C], attention_mask:[B,L])
    input: tuple(input_ids, attention_mask)
    """
    def __init__(self,
                 vocab_size=181,
                 num_hidden_layers=3,
                 feature_dim=768,
                 bidirectional=False,
                 use_pack: bool = True
                 ):
        super(SmilesGRUBackbone, self).__init__()
        self.feature_dim = feature_dim
        self.num_gru_layers = num_hidden_layers
        self.use_pack = use_pack
        self.embedding = nn.Embedding(vocab_size, feature_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=feature_dim,
            num_layers=self.num_gru_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=0.1 if num_hidden_layers > 1 else 0.0)
        gru_out_dim = feature_dim * (2 if bidirectional else 1)
        self.proj_seq = nn.Linear(gru_out_dim, feature_dim)

    def forward(self, input):
        input_ids, attention_mask = input
        input_ids = input_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
        embed = self.embedding(input_ids)
        if self.use_pack:
            # pack_padded_sequence: skip GRU computation for padding tokens
            # note: lengths .cpu() may cause DDP / gpu‑cpu copy overhead
            lengths = attention_mask.sum(dim=1).cpu()
            packed_emb = torch.nn.utils.rnn.pack_padded_sequence(
                embed, lengths, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.gru(packed_emb)
            seq_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=input_ids.size(1))
        else:
            # disable pack: run full padded sequence, avoid cpu‑gpu copy for DDP
            seq_out, _ = self.gru(embed)
        seq_hidden = self.proj_seq(seq_out)
        return seq_hidden, attention_mask


class SmilesLSTMBackbone(nn.Module):
    """
    Unidirectional LSTM-based SMILES backbone, outputs sequence representation
    for outer aggregation. Drop-in replacement for SmilesGRUBackbone.
    forward(input) -> (seq_hidden:[B,L,C], attention_mask:[B,L])
    input: tuple(input_ids, attention_mask)
    """
    def __init__(self,
                 vocab_size=181,
                 num_hidden_layers=3,
                 feature_dim=768,
                 bidirectional=False,
                 use_pack: bool = True
                 ):
        super(SmilesLSTMBackbone, self).__init__()
        self.feature_dim = feature_dim
        self.num_lstm_layers = num_hidden_layers
        self.use_pack = use_pack
        self.embedding = nn.Embedding(vocab_size, feature_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=feature_dim,
            num_layers=self.num_lstm_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=0.1 if num_hidden_layers > 1 else 0.0)
        # unidirectional -> output dim == hidden_size == feature_dim
        # bidirectional -> output dim == 2 * hidden_size == 2 * feature_dim
        gru_out_dim = feature_dim * (2 if bidirectional else 1)
        self.proj_seq = nn.Linear(gru_out_dim, feature_dim)

    def forward(self, input):
        input_ids, attention_mask = input
        input_ids = input_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
        embed = self.embedding(input_ids)
        if self.use_pack:
            # pack_padded_sequence: skip LSTM computation for padding tokens
            # note: lengths .cpu() may cause DDP / gpu-cpu copy overhead
            lengths = attention_mask.sum(dim=1).cpu()
            packed_emb = torch.nn.utils.rnn.pack_padded_sequence(
                embed, lengths, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed_emb)   # (out, (h_n, c_n))
            seq_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=input_ids.size(1))
        else:
            # disable pack: run full padded sequence, avoid cpu-gpu copy for DDP
            seq_out, _ = self.lstm(embed)           # (out, (h_n, c_n))
        seq_hidden = self.proj_seq(seq_out)
        return seq_hidden, attention_mask


class SmilesModel(nn.Module):
    """
    Wrapper module: inject pre‑instantiated backbone object
    Backbone protocol: backbone(input) -> (seq_hidden: [B,L,C], attention_mask:[B,L])
    forward(input) -> (feature, score)
    """
    def __init__(self,
                 backbone: nn.Module,
                 smile_use_esa=False,
                 feature_dim=768,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.smile_use_esa = smile_use_esa
        self.feature_dim = feature_dim

        if self.smile_use_esa:
            self.esa = ESA_SMILES(feature_dim, feature_dim)
        else:
            self.dense_cls = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, feature_dim)
            )

    def forward(self, input):
        seq_hidden, attn_mask = self.backbone(input)
        if self.smile_use_esa:
            smi_feature, smi_score = self.esa(seq_hidden, attn_mask)
        else:
            cls_emb = seq_hidden[:, 0, :]
            smi_feature = self.dense_cls(cls_emb)
            smi_score = None
        return smi_feature, smi_score

    def freeze_backbone(self):
        if hasattr(self.backbone, "freeze_backbone"):
            self.backbone.freeze_backbone()

    def unfreeze_backbone(self):
        if hasattr(self.backbone, "unfreeze_backbone"):
            self.backbone.unfreeze_backbone()


class Attention(nn.Module):
    def __init__(self, n_embed, n_heads, dropout):
        super().__init__()
        head_size = n_embed // n_heads
        self.key = nn.Linear(n_embed, head_size)
        self.query = nn.Linear(n_embed, head_size)
        self.value = nn.Linear(n_embed, head_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ms, ms_mask):
        x = ms
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        wei = q @ k.transpose(-1, -2) / (C ** 0.5)
        mask = ms_mask.unsqueeze(1)
        wei = wei.masked_fill(mask == 0, -1e4)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
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
        out = self.dropout(self.proj(out))
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
        self.sa = MultiHead(n_embed, n_heads, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln = nn.LayerNorm(n_embed)

    def forward(self, ms, ms_mask):
        x = self.ln(ms + self.sa(ms, ms_mask))
        x = self.ln1(x + self.ffwd(x))
        return x, ms_mask


class MSEmbedder(nn.Module):
    """
    PointNovo‑style earliest transformer‑ms embedding, modified based on original MSEmbedder class signature.
    Note: m/z must be pre‑binned to integer bin index in dataloader.
    """
    def __init__(self, max_mz, n_embed, dropout):
        super(MSEmbedder, self).__init__()
        self.mz_embed = nn.Embedding(max_mz, n_embed, padding_idx=0)
        self.intensity_proj = nn.Linear(1, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        mz, intensity, ms_mask = input[0].squeeze(1), input[1].squeeze(1).unsqueeze(-1), input[2].squeeze(1)
        mz_embed = self.mz_embed(mz.long())
        int_embed = self.intensity_proj(intensity)
        x = mz_embed + int_embed
        output = self.dropout(x)
        return output, ms_mask


# class MSEmbedder(nn.Module):
#     def __init__(self, max_mz, n_embed, dropout):
#         super(MSEmbedder, self).__init__()
#         self.mz_embed = nn.Embedding(max_mz, n_embed, padding_idx=0)  

#         self.ff = nn.Sequential(nn.Linear(n_embed + 1, n_embed),
#                                 nn.ReLU(inplace=True),                           
#                                 nn.Linear(n_embed, n_embed),
#                                 nn.Dropout(dropout))
#         self.d_model = n_embed

#     def forward(self, input):
#         mz, intensity, ms_mask = input[0].squeeze(1), input[1].squeeze(1).unsqueeze(-1), input[2].squeeze(1)  # mz (B, 100) i (B, 100) ms_mask (B, 100)
#         mz_embed = self.mz_embed(mz.long()) # (B, 100, 768)
#         x = torch.cat([intensity, mz_embed], dim=2)  # tensor(B, 100,768) ---> (B, 100, 769)
#         output = self.ff(x)
#         return output, ms_mask  # output (1, 101, 768)


class SinusoidalEmbedder(nn.Module):
    def __init__(self, n_embed, dropout):
        super(SinusoidalEmbedder, self).__init__()
        self.mz_embed = nn.Sequential(nn.Linear(n_embed, n_embed),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(n_embed, n_embed))
        self.ff = nn.Sequential(nn.Linear(n_embed + 1, n_embed),
                                nn.ReLU(inplace=True),
                                nn.Linear(n_embed, n_embed),
                                nn.Dropout(dropout))
        self.d_model = n_embed

    def forward(self, input):
        mz, intensity, ms_mask = input[0].squeeze(1), input[1].squeeze(1).unsqueeze(-1), input[2].squeeze(1)
        B, N = ms_mask.shape
        device = ms_mask.device
        result = torch.ones((B, N, self.d_model), device=device)
        pos = mz.unsqueeze(2)
        i = torch.arange(0, self.d_model, 2, device=device)
        div = 10000 ** (i / self.d_model)
        term = pos / div
        result[:, :, 0::2] = torch.sin(term)
        result[:, :, 1::2] = torch.cos(term)
        mz_embed = self.mz_embed(result)
        x = torch.cat([intensity, mz_embed], dim=2)
        output = self.ff(x)
        return output, ms_mask


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


class MSTransformerBackbone(nn.Module):
    """
    Transformer MS backbone
    forward(idx) -> (ms_seq:[B,L,C], ms_mask:[B,L])
    """
    def __init__(self,
                 ms_layer, ms_n_embed, ms_n_heads, ms_dropout,
                 ms_use_se, ms_use_gradient_checkpoint
                 ):
        super().__init__()
        self.ms_use_se = ms_use_se
        self.use_gradient_checkpoint = ms_use_gradient_checkpoint
        n_embed = ms_n_embed
        if self.ms_use_se:
            self.pos_emb = SinusoidalEmbedder(n_embed, ms_dropout)
        else:
            self.pos_emb = MSEmbedder(1660, n_embed, ms_dropout)
        self.blocks = nn.ModuleList([Block(n_embed, ms_n_heads, ms_dropout) for _ in range(ms_layer)])

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
        return ms, ms_mask


class MlpBlock(nn.Module):
    """Residual MLP block, operates point‑wise over sequence dimension [B, L, C]"""
    def __init__(self, channels):
        super(MlpBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        # x shape: [B, L, C], apply mlp to every position independently
        return x + self.block(x)


class CnnBlock2(nn.Module):
    """
    Large‑kernel 1D conv block (single input channel).
    Maps a 1‑D signal [B, L] -> [B, channels, L_out] via Conv1d(1, channels, k, s, p).
    """
    def __init__(self, channels, kernel_size=300, stride=200, padding=200):
        super().__init__()
        self.layer = nn.Conv1d(1,
                               channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)

    def forward(self, x):
        # x: [B, L]  ->  [B, 1, L]  ->  [B, channels, L_out]
        return self.layer(x.unsqueeze(1))


class MSCNNMLPBackbone(nn.Module):
    """
    CNN‑based MS backbone rebuilt from the user's CnnBlock2 + MlpBlock design.

    Two embedding decisions vs. the standalone pasted version:
      1) Layer count is UNIFIED with the transformer backbone via `ms_layer`
         (here it controls the number of residual MlpBlocks after the conv).
      2) Output stays in (B, L, C) form so it plugs straight into the MSModel
         wrapper (ESA / MSProjection) and ModelCLR. We therefore DO NOT flatten;
         the conv output [B, channels, L_out] is permuted to [B, L_out, channels]
         and a per‑position linear projects channels -> feature_dim, so the final
         C equals the shared `feature_dim` used everywhere else.

    forward(idx) -> (ms_seq:[B,L,C], ms_mask:[B,L])
    idx: tuple(mz:[B,1,L], intensity:[B,1,L], ms_mask:[B,1,L])
    """
    def __init__(self,
                 ms_layer=3,
                 cnn_base_channel=64,
                 feature_dim=768,
                 kernel_size=300,
                 stride=200,
                 padding=200,
                 dropout=0.0):
        super().__init__()
        self.channels = cnn_base_channel
        self.feature_dim = feature_dim

        self.conv = CnnBlock2(cnn_base_channel, kernel_size, stride, padding)

        # residual MLP blocks; depth controlled by the unified ms_layer variable
        layers = []
        for _ in range(ms_layer):
            layers.append(MlpBlock(cnn_base_channel))
            layers.append(nn.ReLU())
        self.mlp_stack = nn.Sequential(*layers)

        # per‑position projection to the shared feature_dim -> keeps output C == feature_dim
        self.final_proj = nn.Linear(cnn_base_channel, feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx):
        mz, intensity, ms_mask = idx[0].squeeze(1), idx[1].squeeze(1), idx[2].squeeze(1)
        x = intensity                                   # [B, L]
        x = self.conv(x)                                # [B, channels, L_out]
        x = x.permute(0, 2, 1)                          # [B, L_out, channels]  -> (B, L, C)
        x = self.mlp_stack(x)                           # [B, L_out, channels]
        x = self.dropout(x)
        x = self.final_proj(x)                          # [B, L_out, feature_dim]

        # conv output length is fixed (padding handles edges) -> all‑ones mask of length L_out
        B, L_out = x.size(0), x.size(1)
        out_mask = torch.ones(B, L_out, dtype=torch.long, device=x.device)
        return x, out_mask


class MSModel(nn.Module):
    """
    MS wrapper module: inject pre‑instantiated ms backbone object
    All valid backbones return tuple (ms_seq, ms_mask).
    forward(idx) -> (ms_feature, ms_score)
    """
    def __init__(self,
                 backbone: nn.Module,
                 ms_use_esa=False,
                 feature_dim=768,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.ms_use_esa = ms_use_esa
        self.feature_dim = feature_dim

        if self.ms_use_esa:
            self.esa = ESA_MS(feature_dim, feature_dim)
        else:
            self.projection = MSProjection(feature_dim)

    def forward(self, idx):
        ms_seq, ms_mask = self.backbone(idx)
        if self.ms_use_esa:
            ms_feature, ms_score = self.esa(ms_seq, ms_mask)
        else:
            ms_feature = self.projection(ms_seq)
            ms_score = None
        return ms_feature, ms_score


class ModelCLR(nn.Module):
    """Top‑level contrastive learning model, receives pre‑built SmilesModel and MSModel, pure composition design"""
    def __init__(self,
                 smiles_model: SmilesModel,
                 ms_model: MSModel,
                 ):
        super(ModelCLR, self).__init__()
        self.smiles_model = smiles_model
        self.ms_model = ms_model

    def smiles_encoder(self, xis):
        smi_feature, smi_score = self.smiles_model(xis)
        return smi_feature, smi_score

    def ms_encoder(self, xls):
        ms_feature, ms_score = self.ms_model(xls)
        return ms_feature, ms_score

    def forward(self, smi, ms):
        smi_feature, smi_score = self.smiles_encoder(smi)
        ms_feature, ms_score = self.ms_encoder(ms)
        return smi_feature, smi_score, ms_feature, ms_score


def build_model_from_config(cfg_model: dict) -> ModelCLR:
    """
    Factory function: build full ModelCLR instance from model_config dict in json config file
    """
    # ========== SMILES branch ==========
    smiles_backbone_type = cfg_model.get("smiles_backbone_type", "roberta")
    feature_dim = cfg_model["feature_dim"]
    smile_use_esa = cfg_model["smile_use_esa"]
    vocab_size=cfg_model["vocab_size"]
    num_hidden_layers=cfg_model["num_hidden_layers"]
    smi_dropout=cfg_model["smi_dropout"]

    if smiles_backbone_type == "roberta":
        smiles_backbone = SmilesRobertaBackbone(
            roberta_model_path=cfg_model["roberta_model_path"],
            vocab_size=vocab_size,
            max_position_embeddings=cfg_model["max_position_embeddings"],
            num_attention_heads=cfg_model["num_attention_heads"],
            num_hidden_layers=num_hidden_layers,
            type_vocab_size=cfg_model["type_vocab_size"],
            smi_dropout=smi_dropout,
            feature_dim=feature_dim,
            smiles_use_gradient_checkpoint=cfg_model["smiles_use_gradient_checkpoint"]
        )
    elif smiles_backbone_type == "gru":
        smiles_backbone = SmilesGRUBackbone(
            vocab_size=vocab_size,
            num_hidden_layers=num_hidden_layers,
            feature_dim=feature_dim,
            bidirectional=cfg_model.get("bidirectional", False),
            use_pack=cfg_model.get("use_pack", True)
        )
    elif smiles_backbone_type == "lstm":
        smiles_backbone = SmilesLSTMBackbone(
            vocab_size=vocab_size,
            num_hidden_layers=num_hidden_layers,
            feature_dim=feature_dim,
            bidirectional=cfg_model.get("bidirectional", False),
            use_pack=cfg_model.get("use_pack", True)
        )
    else:
        raise NotImplementedError(f"Unsupported smiles_backbone_type: {smiles_backbone_type}")

    smiles_model = SmilesModel(
        backbone=smiles_backbone,
        smile_use_esa=smile_use_esa,
        feature_dim=feature_dim
    )

    # ========== MS branch ==========
    ms_backbone_type = cfg_model.get("ms_backbone_type", "transformer")
    ms_use_esa = cfg_model["ms_use_esa"]
    ms_layer=cfg_model["ms_layer"]
    ms_n_embed=cfg_model["ms_n_embed"]
    ms_dropout=cfg_model["ms_dropout"]

    if ms_backbone_type == "transformer":
        ms_backbone = MSTransformerBackbone(
            ms_layer=ms_layer,
            ms_n_embed=ms_n_embed,
            ms_n_heads=cfg_model["ms_n_heads"],
            ms_dropout= ms_dropout,
            ms_use_se=cfg_model["ms_use_se"],
            ms_use_gradient_checkpoint=cfg_model["ms_use_gradient_checkpoint"]
        )

    elif ms_backbone_type == "cnn_mlp":
        # rebuilt CNN backbone (CnnBlock2 + residual MlpBlocks), layer depth unified by ms_layer
        ms_backbone = MSCNNMLPBackbone(
            ms_layer=ms_layer,
            cnn_base_channel=cfg_model.get("cnn_base_channel", 64),
            feature_dim=ms_n_embed,
            kernel_size=cfg_model.get("cnn_kernel_size", 300),
            stride=cfg_model.get("cnn_stride", 200),
            padding=cfg_model.get("cnn_padding", 200),
            dropout=ms_dropout
        )

    ms_model = MSModel(
        backbone=ms_backbone,
        ms_use_esa=ms_use_esa,
        feature_dim=ms_n_embed
    )

    model = ModelCLR(smiles_model=smiles_model, ms_model=ms_model)
    return model
