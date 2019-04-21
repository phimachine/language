import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules import LayerNorm
import numpy as np

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(0)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).float().unsqueeze(-1)

def from_pretrained(embeddings, freeze=True):
    assert embeddings.dim() == 2, \
         'Embeddings parameter is expected to be 2-dimensional'
    rows, cols = embeddings.shape
    embedding = torch.nn.Embedding(num_embeddings=rows, embedding_dim=cols)
    embedding.weight = torch.nn.Parameter(embeddings)
    embedding.weight.requires_grad = not freeze
    return embedding


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


class TransformerBOW(nn.Module):
    # no attention module here, because the input is not timewise
    # we will add time-wise attention to time-series model later.

    def __init__(self, d_model=256, d_inner=32, dropout=0.1, n_layers=8, vocab_size=50000, output_size=12):
        super(TransformerBOW, self).__init__()

        self.first_embedding=nn.Linear(vocab_size,d_model)

        self.layer_stack = nn.ModuleList([
            EncoderLayerBOW(d_model, d_inner, dropout=dropout)
            for _ in range(n_layers)])

        self.last_linear=nn.Linear(d_model,output_size)

    @staticmethod
    def reset_mod(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.reset_parameters()

    def reset_parameters(self):
        self.apply(self.reset_mod)

    def forward(self, input):
        """

        :param input: (batch_size, embedding)
        :return:
        """
        enc_output=self.first_embedding(input)
        enc_output=enc_output.unsqueeze(1)
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output)

        output=self.last_linear(enc_output)
        output=output.squeeze(1)
        return output

class Transformer(nn.Module):
    """
    The time-wise transformer model
    """
    def __init__(self, d_model=64, d_inner=16, n_head=8, d_k=16, d_v=16, dropout=0.1, n_layers=4, vocab_size=50000,
                 output_size=12, max_len=1000):
        super(Transformer, self).__init__()

        self.max_len=max_len
        self.d_inner=d_inner
        self.n_head=n_head
        self.d_k=d_k
        self.d_v=d_v
        self.dropout=dropout
        self.n_layer=n_layers
        self.vocab_size=vocab_size


        self.first_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_enc=from_pretrained(get_sinusoid_encoding_table(max_len+1, d_model, padding_idx=0), freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        self.last_linear = nn.Linear(d_model, output_size)


    @staticmethod
    def reset_mod(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.reset_parameters()

    def reset_parameters(self):
        self.apply(self.reset_mod)

    def forward(self, input):
        """

        :param input: (batch_size, embedding)
        :return:
        """

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=input, seq_q=input)
        non_pad_mask = get_non_pad_mask(input)

        # -- Forward
        enc_output = self.first_embedding(input) + self.position_enc(input)

        for enc_layer in self.layer_stack:
            enc_output, attn = enc_layer(enc_output,non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)

        output = self.last_linear(enc_output)
        output = output.max(1)[0]
        return output


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model=512, d_inner=512, n_head=8, d_k=64, d_v=64, dropout=0.1):
        super(EncoderLayer, self).__init__()
        # self attention. see forward. query, key, value are the same
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        # self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.fc=FullFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn

class EncoderLayerBOW(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model=512, d_inner=512, dropout=0.1):
        super(EncoderLayerBOW, self).__init__()
        # self attention. see forward. query, key, value are the same
        # self.slf_attn = MultiHeadAttentionBOW(
        #     n_head, d_model, d_k, d_v, dropout=dropout)
        # self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.fc=FullFeedForward(d_model, d_inner, dropout=dropout)
    def forward(self, enc_input):
        enc_output=enc_input
        # removing self-attention seems to achieve good results. Why?
        # enc_output, enc_slf_attn = self.slf_attn(
        #     enc_input, enc_input, enc_input, mask=None)

        enc_output = self.fc(enc_output)

        return enc_output



class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        # softmax over the time steps
        # each time step embedding has a softmax
        # so (time_steps, time_steps)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class MultiHeadAttentionBOW(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        # mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        # softmax over the time steps
        # each time step embedding has a softmax
        # so (time_steps, time_steps)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class FullFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        # output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(x)))
        # output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e12)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn
