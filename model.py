import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from einops import rearrange


class PositionEncoding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even."
        # 由于位置编码是一个二元函数，可以考虑使用一个二维的矩阵来表示
        i_pos = torch.linspace(
            0, max_seq_len - 1, max_seq_len
        )  # [0, 1, 2, ..., max_len-1] 表示 pos
        j_dim = torch.linspace(
            0, d_model - 2, d_model // 2
        )  # [0, 2, 4, ..., d_model-2] 表示偶数的 dim

        # 生成一个 [max_len, d_model//2] 的网格网格
        # pos.shape: [max_len, d_model//2], two_i.shape: [max_len, d_model//2]
        pos, two_i = torch.meshgrid(i_pos, j_dim, indexing="ij")
        # pe_two_i.shape: [max_len, d_model//2]
        pe_two_i = torch.sin(pos / (10000 ** (two_i / d_model)))
        # pe_two_i_1.shape: [max_len, d_model//2]
        pe_two_i_1 = torch.cos(pos / (10000 ** (two_i / d_model)))

        # 将 pe_two_i 和 pe_two_i_1 拼接成一个 [max_len, d_model] 的矩阵
        # 考虑先拼成 [max_len, d_model//2, 2] 的矩阵，然后再展平
        # 利用 torch.stack 将两个 tensor 沿着最后一个维度堆叠 (若使用 torch.cat 的话则直接连接了，不符合偶奇间隔的要求)
        pe = torch.stack([pe_two_i, pe_two_i_1], dim=-1)  # [max_len, d_model//2, 2]
        pe = pe.reshape(
            1, max_seq_len, d_model
        )  # [1, max_len, d_model] 预留出 batch 维度
        # 注册为 buffer，即不需要更新的参数 (需要更新的参数为 parameter)
        # 当我们使用 model.to(device) 时，buffer 和 parameter 都会自动转移到对应的设备上
        self.register_buffer("pe", pe, False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            x: [batch_size, seq_len, d_model]
        """
        _, seq_len, d_model = x.shape
        assert seq_len <= self.pe.shape[1], "seq_len exceeds max_len."
        assert d_model == self.pe.shape[2], "d_model mismatch."
        # 直接加上位置编码
        return x + self.pe[:, :seq_len, :]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % heads == 0, "d_model must be divisible by heads."
        self.d_model = d_model
        self.heads = heads
        self.dim_head = d_model // heads
        inner_dim = heads * self.dim_head
        self.WQ = nn.Linear(d_model, inner_dim)
        self.WK = nn.Linear(d_model, inner_dim)
        self.WV = nn.Linear(d_model, inner_dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(inner_dim, d_model)
        self.INF = float(1e12)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            q: [batch_size, q_len, d_model]
            k: [batch_size, k_len, d_model]
            v: [batch_size, k_len, d_model]
        Returns:
            out: [batch_size, seq_len, d_model]
        """

        assert q.shape[0] == k.shape[0] == v.shape[0], "batch size mismatch."
        assert k.shape[1] == v.shape[1], "key and value length mismatch."
        # [batch_size, len, d_model] -> [batch_size, len, inner_dim] -> [batch_size, len, heads, dim_head] -> [batch_size, heads, len, dim_head]
        Q = rearrange(self.WQ(q), "b l (h d) -> b h l d", h=self.heads)
        K = rearrange(self.WK(k), "b l (h d) -> b h l d", h=self.heads)
        V = rearrange(self.WV(v), "b l (h d) -> b h l d", h=self.heads)
        dots = torch.matmul(Q, K.transpose(-2, -1)) / (
            self.dim_head**0.5
        )  # [batch_size, heads, q_len, k_len]
        if mask is not None:
            # mask 为判断条件，将 mask 为 True 的部分填充为自定义的 s-inf
            # 如果直接使用 python 自带的 inf 进行填充的话会得到 nan
            dots.masked_fill_(mask, -self.INF)
        attn = self.attend(dots)
        out = rearrange(
            torch.matmul(attn, V), "b h l d -> b l (h d)"
        )  # [batch_size, q_len, inner_dim]
        out = self.fc(self.dropout(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            out: [batch_size, seq_len, d_model]
        """
        out = self.linear1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.ln1(x + self.dropout1(self.self_attention(x, x, x, mask)))
        x = self.ln2(x + self.dropout2(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.cross_attention = MultiHeadAttention(d_model, heads, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_kv: torch.Tensor,
        dst_mask: Optional[torch.Tensor] = None,
        src_dst_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.ln1(x + self.dropout1(self.self_attention(x, x, x, dst_mask)))
        x = self.ln2(
            x
            + self.dropout2(
                self.cross_attention(x, encoder_kv, encoder_kv, src_dst_mask)
            )
        )
        x = self.ln3(x + self.dropout3(self.ffn(x)))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        d_model: int,
        d_ff: int,
        n_layers: int,
        heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        # pad 不参与梯度计算
        self.embedding = nn.Embedding(vocab_size, d_model, pad_idx)
        self.position_encoding = PositionEncoding(max_seq_len, d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask: Optional[torch.Tensor] = None):
        x = self.embedding(x)
        x = self.position_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        d_model: int,
        d_ff: int,
        n_layers: int,
        heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, pad_idx)
        self.position_encoding = PositionEncoding(max_seq_len, d_model)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_kv,
        dst_mask: Optional[torch.Tensor] = None,
        src_dst_mask: Optional[torch.Tensor] = None,
    ):
        x = self.embedding(x)
        x = self.position_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, encoder_kv, dst_mask, src_dst_mask)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        dst_vocab_size: int,
        pad_idx: int,
        d_model: int,
        d_ff: int,
        n_layers: int,
        heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.encoder = Encoder(
            src_vocab_size,
            pad_idx,
            d_model,
            d_ff,
            n_layers,
            heads,
            dropout,
            max_seq_len,
        )
        self.decoder = Decoder(
            dst_vocab_size,
            pad_idx,
            d_model,
            d_ff,
            n_layers,
            heads,
            dropout,
            max_seq_len,
        )
        self.pad_idx = pad_idx
        self.output_layer = nn.Linear(d_model, dst_vocab_size)

    def generate_mask(
        self, q_pad: torch.Tensor, k_pad: torch.Tensor, apply_causal_mask: bool = False
    ):
        # q_pad shape: [n, q_len]
        # k_pad shape: [n, k_len]
        # q_pad k_pad dtype: bool
        assert q_pad.device == k_pad.device, "padding mask must be same device."
        n, q_len = q_pad.shape
        n, k_len = k_pad.shape

        mask_shape = (n, 1, q_len, k_len)
        if apply_causal_mask:
            # Decoder mask
            mask = 1 - torch.tril(
                torch.ones(mask_shape)
            )  # 对角线以上全为 1，即屏蔽之前的信息

        else:
            # Encoder mask
            mask = torch.zeros(mask_shape)
        mask = mask.to(q_pad.device)
        for i in range(n):
            mask[i, :, q_pad[i], :] = 1
            mask[i, :, :, k_pad[i]] = 1
        mask = mask.to(torch.bool)
        return mask

    def forward(self, x, y):
        src_pad_mask = x == self.pad_idx
        dst_pad_mask = y == self.pad_idx
        src_mask = self.generate_mask(
            q_pad=src_pad_mask, k_pad=src_pad_mask, apply_causal_mask=False
        )  # 使用 encoder mask
        dst_mask = self.generate_mask(
            q_pad=dst_pad_mask, k_pad=dst_pad_mask, apply_causal_mask=True
        )  # 使用 decoder mask
        src_dst_mask = self.generate_mask(
            q_pad=dst_pad_mask, k_pad=src_pad_mask, apply_causal_mask=False
        )
        encoder_kv = self.encoder(x, src_mask)
        res = self.decoder(y, encoder_kv, dst_mask, src_dst_mask)
        res = self.output_layer(res)
        return res
