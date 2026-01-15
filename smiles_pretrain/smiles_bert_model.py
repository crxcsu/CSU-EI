#  smiles encoder 代码
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# position embedding
# class PositionalEmbedding(nn.Module):

#     def __init__(self, hidden_size, max_len=512):
#         super().__init__()

#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, hidden_size).float()
#         pe.require_grad = False

#         position = torch.arange(0, max_len).float().unsqueeze(1)
#         div_term = (torch.arange(0, hidden_size, 2).float() * -(math.log(10000.0) / hidden_size)).exp()

#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)

#         # 对数据维度进行扩充，扩展第0维
#         pe = pe.unsqueeze(0)
#         # 添加一个持久缓冲区pe,缓冲区可以使用给定的名称作为属性访问
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return self.pe[:, :x.size(1)]

# embedding 层
class Embedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, pad_token_id):
        super(Embedding, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        # 位置编码作为可学习参数
        self.position_embeddings = nn.Parameter(torch.randn(1, max_position_embeddings, hidden_size))

    def forward(self, input_ids):
        token_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings()
        embeddings = token_embeddings + position_embeddings.data[:, :input_ids.shape[1], :] # [batch_size, seq_length, hidden_size]
        return embeddings

# self attention 层
class BertSelfAttention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, attention_probs_dropout_prob):
        super(BertSelfAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_ids, attention_mask):
        mixed_query_layer = self.query(input_ids) # [batch_size, seq_length, hidden_size]
        mixed_key_layer = self.key(input_ids)
        mixed_value_layer = self.value(input_ids)
        # [batch_size, seq_length, hidden_size] -> [batch_size, num_attention_heads, seq_length, attention_head_size]
        # 实现多头注意力机制
        query_layer = self.transpose_for_scores(mixed_query_layer) # [batch_size, num_attention_heads, seq_length, attention_head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # 注意力分数 QK^T
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # attention_mask
        mask = (1.0 - attention_mask) * -1000000.0 
        attention_scores = attention_scores + mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        # 点乘V
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

# feed forward 层
class BertIntermediate(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(BertIntermediate).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(hidden_dropout_prob)
        )

    def forward(self, x):
        return self.ffn(x)

# bert layer
class BertLayer(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, attention_probs_dropout_prob, hidden_dropout_prob):
        super(BertLayer).__init__()
        self.attention = BertSelfAttention(num_attention_heads, hidden_size, attention_probs_dropout_prob)
        self.intermediate = BertIntermediate(hidden_size, hidden_dropout_prob)
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, input_ids, attention_mask):
        attention_output = self.layer_norm1(self.attention(input_ids, attention_mask) + input_ids) # 残差连接
        intermediate_output = self.layer_norm2(self.intermediate(attention_output) + attention_output)
        return intermediate_output

class BertEncoder(nn.Module):
    def __init__(self, vocab_size, num_attention_heads, hidden_size, attention_probs_dropout_prob, hidden_dropout_prob, num_hidden_layers):
        super(BertEncoder).__init__()
        self.embed = Embedding(vocab_size, hidden_size, )
        self.layers = nn.ModuleList([BertLayer(num_attention_heads, hidden_size, attention_probs_dropout_prob, hidden_dropout_prob) for _ in range(num_hidden_layers)])


# feed forward 层
class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask):
        hidden_states = self.dense(input_ids)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
