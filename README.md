# 基于 Transformer 的机器翻译（英译中）


## 📋 项目背景


### 1. 机器翻译的技术演进
机器翻译 (Machine Translation, MT) 作为自然语言处理 (NLP) 领域的核心任务之一, 经历了三个主要发展阶段: 
- **规则驱动时代** (1950s-1990s): 基于语言学专家制定的语法规则和双语词典进行直译, 受限于语言复杂性难以实现流畅翻译
- **统计学习时代** (2000s-2010s): IBM 提出的基于短语的统计机器翻译 (SMT) 成为主流, 利用大规模双语语料库学习翻译概率模型
- **神经网络时代** (2017-至今): 2017 年 Google 提出的 Transformer 架构引发革命, 其自注意力机制突破了传统 RNN 的序列建模瓶颈

### 2. 英译中任务的特殊挑战
中文与英语的跨语种翻译存在多重难点: 
- **结构差异**: 英语的 SVOC (主谓宾补) 结构与中文的意合语法存在映射鸿沟
- **语义鸿沟**: 成语 (如 "画蛇添足")、文化专有项 (如 "红包") 的等效表达问题
- **数据稀缺性**: 高质量英中平行语料规模仅为英法双语数据的 1/5 (WMT 2020 统计)

### 3. Transformer 的技术优势
本项目选用 Transformer 架构的核心理由: 
| 特性                | RNN/LSTM          | Transformer       |
|---------------------|-------------------|-------------------|
| 长距离依赖建模       | 随距离衰减         | 全局注意力        |
| 训练并行度           | 序列逐步计算       | 全序列并行        |
| 计算复杂度           | O(n)              | O(n²)            |
| 位置敏感性           | 固有顺序          | 需位置编码        |


## 🛠️ 安装

### 环境配置

```bash
conda create -n translator python=3.10
conda activate translator
pip install -r requirements.txt
```

### 数据集配置

本项目使用 [cmn-eng-simple](https://box.nju.edu.cn/d/b8245873f1e44c9fab65/) 数据集, 包含英中平行语料. 通过链接下载后, 将其放置在 `data` 目录, 结构如下: 

```
data/
└── cmn-eng-simple/
    ├── training.txt       
    ├── validation.txt     
    ├── testing.txt       
    ├── int2word_cn.json   
    ├── word2int_cn.json  
    ├── int2word_en.json  
    └── word2int_en.json  
```

## 🚀 快速运行

``` bash
python main.py --period train
python main.py --period eval
```

## 📊 实验报告

### 模型搭建

Transformer 模型结构如下图所示. 其中, 主要模块包括左侧的编码器（Encoder）, 以及右侧的解码器（Decoder）两个部分. 编码器将输入序列转换为上下文向量, 解码器根据上下文向量生成目标序列. 

<img src="figs/transformer.png" height="600">

我们先从小的组件开始实现, 最后再将它们组合成完整的 Transformer 模型. 

1. Input Embedding

    在将自然语言输入模型前, 我们首先会对其进行分词, 再根据词汇表将每个词根转换为对应的 token. 例如 `i am a student .` 会被转化为 `[5, 98, 9, 415, 4]`. 这样自然语言就变成了计算机可以理解的数值形式. 当然, 对于深度学习模型来说, 这还不够. 现在每个 token 还是处于文本空间当中, 我们希望将其投影到模型的语义空间, 以便模型更好地理解和处理其特征. PyTorch 已经提供了一个模块 `torch.nn.Embedding` 用于该操作. 初始化时, 与机器翻译相关的参数包括

    - 词汇表大小: `num_embeddings (int)`
    - 特征维度: `embedding_dim (int)`
    - 填充标记: `padding_idx (int, optional)`

    在构建最终的模型时, 我们直接使用即可. 

2. Positional Encoding

    在自然语言中, 单词的顺序是非常重要的. 为了让模型能够理解单词的顺序, 我们需要为每个单词添加一个位置编码. 位置编码是一个与单词嵌入（word embedding）相同维度的向量, 它包含了单词在句子中的位置信息. 论文中给出的位置编码公式如下: 

    $PE_{(pos, 2i)} = sin(pos / 10000^{2i/d_{model}})$

    $PE_{(pos, 2i+1)} = cos(pos / 10000^{2i/d_{model}})$

    其中, $pos$ 是单词在句子中的位置, $i$ 是位置编码的维度索引, $d_{model}$ 是嵌入向量的维度. 我们可以直接将其理解为一个编码层, Transformer 的输入首先通过该编码层, 获得位置编码. 代码实现如下: 

    ```python
    class PositionEncoding(nn.Module):
        def __init__(self, max_seq_len: int, d_model: int):
            super().__init__()
            assert d_model % 2 == 0, "d_model must be even."
            # 由于位置编码是一个二元函数, 可以考虑使用一个二维的矩阵来表示
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
            # 考虑先拼成 [max_len, d_model//2, 2] 的矩阵, 然后再展平
            # 利用 torch.stack 将两个 tensor 沿着最后一个维度堆叠 (若使用 torch.cat 的话则直接连接了, 不符合偶奇间隔的要求)
            pe = torch.stack([pe_two_i, pe_two_i_1], dim=-1)  # [max_len, d_model//2, 2]
            pe = pe.reshape(
                1, max_seq_len, d_model
            )  # [1, max_len, d_model] 预留出 batch 维度
            # 注册为 buffer, 即不需要更新的参数 (需要更新的参数为 parameter)
            # 当我们使用 model.to(device) 时, buffer 和 parameter 都会自动转移到对应的设备上
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
    ```

3. Multi-Head Attention

    在 Transformer 中, 注意力机制是一个非常重要的组成部分. 它允许模型在处理输入序列时, 关注序列中的不同部分. Multi-Head Attention 是一种将多个注意力头（attention head）结合起来的方法. 每个注意力头都有自己的参数集, 可以学习到不同的特征表示. 最终的输出是所有注意力头的输出拼接在一起. 

    <img src="figs/attention.png" height="300">

    其公式如下所示: 

    $\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$

    其中, $Q, K, V$ 分别表示查询 (Query)、键 (Key) 和值 (Value), $h$ 表示头数, $W^O$ 是一个线性变换矩阵. 每个头的计算公式为: 

    $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

    其中, $W_i^Q$, $W_i^K$, $W_i^V$ 是每个头的线性变换矩阵. 注意力机制的计算公式为: 

    $\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$


    其中, $d_k$ 是键的维度. 
    
    Multi-Head Attention 的核心代码实现如下: 

    ```python
    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model: int, heads: int, dropout: float = 0.1):
            super().__init__()

            ...
            self.INF = float(1e12)

        def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            Q = rearrange(self.WQ(q), "b l (h d) -> b h l d", h=self.heads)
            K = rearrange(self.WK(k), "b l (h d) -> b h l d", h=self.heads)
            V = rearrange(self.WV(v), "b l (h d) -> b h l d", h=self.heads)
            dots = torch.matmul(Q, K.transpose(-2, -1)) / (
                self.dim_head**0.5
            )  # [batch_size, heads, q_len, k_len]
            if mask is not None:
                dots.masked_fill_(mask, -self.INF)
            attn = self.attend(dots)
            out = rearrange(
                torch.matmul(attn, V), "b h l d -> b l (h d)"
            )  # [batch_size, q_len, inner_dim]
            out = self.fc(self.dropout(out))
            return out
    ```
    > 需要特别注意的是, 此处在设置 mask 时, 我们没有直接使用 Python 中自带的 inf, 而是设置了一个自定义的大数. 若直接使用 inf, 在计算 Softmax 后, 会出现对应位置值为 nan 的情况. 

4. Feed Forward

    Feed Forward 是 Transformer 中的一个重要组成部分. 它是一个两层的前馈神经网络, 通常使用 ReLU 激活函数. 其公式如下: 

    $\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$
    
    其中, $W_1, b_1$ 是第一层的权重和偏置, $W_2, b_2$ 是第二层的权重和偏置. Feed Forward 的代码实现如下: 

    ```python
    class FeedForward(nn.Module):
        def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
            super().__init__()
            self.linear1 = nn.Linear(d_model, d_ff)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(d_ff, d_model)
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.linear1(x)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.linear2(out)
            return out
    ```

5. Encoder Layer

    <img src="figs/encoder_layer.png" height="300">

    每个 Encoder Layer 主要由两部分组成: Multi-Head Attention（此处的 Multi-Head Attention 为 Self-Attention） 和 Feed Forward. 它们之间还有一个 Add & Norm 的处理, 即残差连接（Residual Connection）和层归一化（Layer Normalization）. 残差连接允许梯度在反向传播时更容易地流过网络, 从而加速训练. 层归一化用于稳定训练过程. Encoder Layer 的代码实现如下: 

    ```python
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
    ```

6. Decoder Layer

    Decoder Layer 与 Encoder Layer 类似. 但需要注意的是, Decoder Layer 中第一个 Multi-Head Attention 为 Self-Attention, 其中, Q、K、V 均来自 Decoder 的输入；第二个 Multi-Head Attention 则为 Cross-Attention, 其中, Q 来自 Decoder 的输入, K、V 则为 Encoder 的输出. 其代码实现如下: 

    ```python
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
    ```

7. Encoder

    编码器由多个编码器层堆叠而成. 其代码实现如下: 

    ```python
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
    ```

8. Decoder

    解码器由多个解码器层堆叠而成. 其代码实现如下: 

    ```python
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
    ```

有了以上的组件, 我们就可以将它们组合成完整的 Transformer 模型了. Transformer 模型的代码实现如下: 

```python
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
        n, q_len = q_pad.shape
        n, k_len = k_pad.shape

        mask_shape = (n, 1, q_len, k_len)
        if apply_causal_mask:
            # Decoder mask
            mask = 1 - torch.tril(
                torch.ones(mask_shape)
            )  # 对角线以上全为 1, 即屏蔽之前的信息

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
```

除了已有的小组件之外, 可以看到 Transformer 中还包括了一个很重要的函数 `generate_mask`, 该函数用于生成 padding mask 和 causal mask. padding mask 用于屏蔽掉输入序列中的填充部分, 而 causal mask 则用于屏蔽掉解码器中当前 token 之后的部分. 这样可以确保模型在生成下一个 token 时, 只能看到当前 token 之前的部分. 

至此, 模型就已搭建完毕. 

### 数据处理

本次实验的任务是将中文翻译成英文, 使用的数据集包含了 21621 条中英平行语料, 每对语料中包含一条英文输入语句及其对应的中文翻译结果, 英文和中文之间使用制表符 (tab) 进行分隔. 例如: 

```txt
it 's none of your concern . 	这不关 你 的 事 。
```

在数据处理阶段, 我们需要将自然语言转换为计算机可以理解的数值形式, 即 token 序列. 在 `cmn-eng-simple` 数据集当中, 预处理时使用 jieba 分词器给中文文本进行分词, 使用 BPE 分词器给英文进行分词, 并预先定义好了自然语言词语与 token 之间的一对一映射关系 (即词表), 具体可见 `int2word_cn.json`、`int2word_en.json`、`word2int_cn.json`、`word2int_en.json` 四个文件. 其中, 英文词表的大小为 3922, 中文词表的大小为 3775. 我们可以直接使用这些文件来进行数据处理. 为了方便后续的训练和评测时按照 batch 进行处理, 我们同时将数据集封装成了 `TranslationDataset` 类, 继承自 `torch.utils.data.Dataset` 类. 当我们实例化 `TranslationDataset` 类时, 会自动加载词表, 并根据词表将自然语言转换为 token 序列, 并可通过下标访问到转换成 `torch.LongTensor` 类型的英文和中文 token 序列. 具体代码可见 `dataset.py` 文件. 

构建数据集后, 我们可以通过如下方式检验能否正确加载数据集: 

```python
if __name__ == "__main__":
    data_dir = "data/cmn-eng-simple"
    dataset = TranslationDataset(data_dir, split="train")
    print(f"Number of samples: {len(dataset)}\n")
    for i in range(3):
        en, cn = dataset[i]
        print("English: ", end=" ")
        for word in en:
            print(dataset.int2en[str(word.item())], end=" ")
        print("\nChinese: ", end=" ")
        for word in cn:
            print(dataset.int2cn[str(word.item())], end=" ")
        print("\n")
```

输出结果如下所示: 

```txt
Number of samples: 18000

English:  <BOS> it 's none of your concern . <EOS> 
Chinese:  <BOS> 这不关 你 的 事 .  <EOS> 

English:  <BOS> she has a habit of <UNK> ting her na ils . <EOS> 
Chinese:  <BOS> 她 有 咬 <UNK> 的 习惯 .  <EOS> 

English:  <BOS> he is a teacher . <EOS> 
Chinese:  <BOS> 他 是 老师 .  <EOS>  
```
故数据集加载成功. 

### 超参数设置

本次实验中, 我们通过 `config.py` 文件来管理超参数, 具体如下: 

| 参数名         | 默认值         | 说明               |
|----------------|---------------|--------------------|
| seed           | 2025          | 随机种子           |
| d_model        | 512           | 模型内部维度       |
| d_ff           | 2048          | 前馈网络维度       |
| n_layers       | 6             | 编码器和解码器层数 |
| heads          | 8             | 注意力头数         |
| dropout        | 0.1           | dropout 概率       |
| max_seq_len    | 100           | 最大序列长度       |
| batch_size     | 16            | 批次大小           |
| lr             | 1e-4          | 学习率             |
| n_epochs       | 60            | 训练轮数           |
| print_interval | 50            | 打印间隔           |

其中, 随机种子用于保证实验结果的可重复性, 模型内部维度、前馈网络维度、编码器和解码器层数、注意力头数、dropout 概率、最大序列长度、批次大小、学习率、训练轮数、打印间隔等参数则用于控制模型的训练过程和性能. 我们可以打印出模型的结构和参数量: 

```txt
    +-------------------------------------------------------+----------------+------------+-------------+
    | Layer (type)                                          | Output Shape   | Param #    | Trainable   |
    |-------------------------------------------------------+----------------+------------+-------------|
    | encoder (Encoder)                                     | [--]           | 20,922,368 | Yes         |
    | encoder.embedding (Embedding)                         | [--]           | 2,008,064  | Yes         |
    | encoder.layers (ModuleList)                           | [--]           | 18,914,304 | Yes         |
    | encoder.layers.0 (EncoderLayer)                       | [--]           | 3,152,384  | Yes         |
    | encoder.layers.0.self_attention (MultiHeadAttention)  | [--]           | 1,050,624  | Yes         |
    | encoder.layers.0.ffn (FeedForward)                    | [--]           | 2,099,712  | Yes         |
    | encoder.layers.0.ln1 (LayerNorm)                      | [--]           | 1,024      | Yes         |
    | encoder.layers.0.ln2 (LayerNorm)                      | [--]           | 1,024      | Yes         |
    | encoder.layers.1 (EncoderLayer)                       | [--]           | 3,152,384  | Yes         |
    | encoder.layers.1.self_attention (MultiHeadAttention)  | [--]           | 1,050,624  | Yes         |
    | encoder.layers.1.ffn (FeedForward)                    | [--]           | 2,099,712  | Yes         |
    | encoder.layers.1.ln1 (LayerNorm)                      | [--]           | 1,024      | Yes         |
    | encoder.layers.1.ln2 (LayerNorm)                      | [--]           | 1,024      | Yes         |
    | encoder.layers.2 (EncoderLayer)                       | [--]           | 3,152,384  | Yes         |
    | encoder.layers.2.self_attention (MultiHeadAttention)  | [--]           | 1,050,624  | Yes         |
    | encoder.layers.2.ffn (FeedForward)                    | [--]           | 2,099,712  | Yes         |
    | encoder.layers.2.ln1 (LayerNorm)                      | [--]           | 1,024      | Yes         |
    | encoder.layers.2.ln2 (LayerNorm)                      | [--]           | 1,024      | Yes         |
    | encoder.layers.3 (EncoderLayer)                       | [--]           | 3,152,384  | Yes         |
    | encoder.layers.3.self_attention (MultiHeadAttention)  | [--]           | 1,050,624  | Yes         |
    | encoder.layers.3.ffn (FeedForward)                    | [--]           | 2,099,712  | Yes         |
    | encoder.layers.3.ln1 (LayerNorm)                      | [--]           | 1,024      | Yes         |
    | encoder.layers.3.ln2 (LayerNorm)                      | [--]           | 1,024      | Yes         |
    | encoder.layers.4 (EncoderLayer)                       | [--]           | 3,152,384  | Yes         |
    | encoder.layers.4.self_attention (MultiHeadAttention)  | [--]           | 1,050,624  | Yes         |
    | encoder.layers.4.ffn (FeedForward)                    | [--]           | 2,099,712  | Yes         |
    | encoder.layers.4.ln1 (LayerNorm)                      | [--]           | 1,024      | Yes         |
    | encoder.layers.4.ln2 (LayerNorm)                      | [--]           | 1,024      | Yes         |
    | encoder.layers.5 (EncoderLayer)                       | [--]           | 3,152,384  | Yes         |
    | encoder.layers.5.self_attention (MultiHeadAttention)  | [--]           | 1,050,624  | Yes         |
    | encoder.layers.5.ffn (FeedForward)                    | [--]           | 2,099,712  | Yes         |
    | encoder.layers.5.ln1 (LayerNorm)                      | [--]           | 1,024      | Yes         |
    | encoder.layers.5.ln2 (LayerNorm)                      | [--]           | 1,024      | Yes         |
    | decoder (Decoder)                                     | [--]           | 27,156,992 | Yes         |
    | decoder.embedding (Embedding)                         | [--]           | 1,932,800  | Yes         |
    | decoder.layers (ModuleList)                           | [--]           | 25,224,192 | Yes         |
    | decoder.layers.0 (DecoderLayer)                       | [--]           | 4,204,032  | Yes         |
    | decoder.layers.0.self_attention (MultiHeadAttention)  | [--]           | 1,050,624  | Yes         |
    | decoder.layers.0.ln1 (LayerNorm)                      | [--]           | 1,024      | Yes         |
    | decoder.layers.0.cross_attention (MultiHeadAttention) | [--]           | 1,050,624  | Yes         |
    | decoder.layers.0.ln2 (LayerNorm)                      | [--]           | 1,024      | Yes         |
    | decoder.layers.0.ffn (FeedForward)                    | [--]           | 2,099,712  | Yes         |
    | decoder.layers.0.ln3 (LayerNorm)                      | [--]           | 1,024      | Yes         |
    | decoder.layers.1 (DecoderLayer)                       | [--]           | 4,204,032  | Yes         |
    | decoder.layers.1.self_attention (MultiHeadAttention)  | [--]           | 1,050,624  | Yes         |
    | decoder.layers.1.ln1 (LayerNorm)                      | [--]           | 1,024      | Yes         |
    | decoder.layers.1.cross_attention (MultiHeadAttention) | [--]           | 1,050,624  | Yes         |
    | decoder.layers.1.ln2 (LayerNorm)                      | [--]           | 1,024      | Yes         |
    | decoder.layers.1.ffn (FeedForward)                    | [--]           | 2,099,712  | Yes         |
    | decoder.layers.1.ln3 (LayerNorm)                      | [--]           | 1,024      | Yes         |
    | decoder.layers.2 (DecoderLayer)                       | [--]           | 4,204,032  | Yes         |
    | decoder.layers.2.self_attention (MultiHeadAttention)  | [--]           | 1,050,624  | Yes         |
    | decoder.layers.2.ln1 (LayerNorm)                      | [--]           | 1,024      | Yes         |
    | decoder.layers.2.cross_attention (MultiHeadAttention) | [--]           | 1,050,624  | Yes         |
    | decoder.layers.2.ln2 (LayerNorm)                      | [--]           | 1,024      | Yes         |
    | decoder.layers.2.ffn (FeedForward)                    | [--]           | 2,099,712  | Yes         |
    | decoder.layers.2.ln3 (LayerNorm)                      | [--]           | 1,024      | Yes         |
    | decoder.layers.3 (DecoderLayer)                       | [--]           | 4,204,032  | Yes         |
    | decoder.layers.3.self_attention (MultiHeadAttention)  | [--]           | 1,050,624  | Yes         |
    | decoder.layers.3.ln1 (LayerNorm)                      | [--]           | 1,024      | Yes         |
    | decoder.layers.3.cross_attention (MultiHeadAttention) | [--]           | 1,050,624  | Yes         |
    | decoder.layers.3.ln2 (LayerNorm)                      | [--]           | 1,024      | Yes         |
    | decoder.layers.3.ffn (FeedForward)                    | [--]           | 2,099,712  | Yes         |
    | decoder.layers.3.ln3 (LayerNorm)                      | [--]           | 1,024      | Yes         |
    | decoder.layers.4 (DecoderLayer)                       | [--]           | 4,204,032  | Yes         |
    | decoder.layers.4.self_attention (MultiHeadAttention)  | [--]           | 1,050,624  | Yes         |
    | decoder.layers.4.ln1 (LayerNorm)                      | [--]           | 1,024      | Yes         |
    | decoder.layers.4.cross_attention (MultiHeadAttention) | [--]           | 1,050,624  | Yes         |
    | decoder.layers.4.ln2 (LayerNorm)                      | [--]           | 1,024      | Yes         |
    | decoder.layers.4.ffn (FeedForward)                    | [--]           | 2,099,712  | Yes         |
    | decoder.layers.4.ln3 (LayerNorm)                      | [--]           | 1,024      | Yes         |
    | decoder.layers.5 (DecoderLayer)                       | [--]           | 4,204,032  | Yes         |
    | decoder.layers.5.self_attention (MultiHeadAttention)  | [--]           | 1,050,624  | Yes         |
    | decoder.layers.5.ln1 (LayerNorm)                      | [--]           | 1,024      | Yes         |
    | decoder.layers.5.cross_attention (MultiHeadAttention) | [--]           | 1,050,624  | Yes         |
    | decoder.layers.5.ln2 (LayerNorm)                      | [--]           | 1,024      | Yes         |
    | decoder.layers.5.ffn (FeedForward)                    | [--]           | 2,099,712  | Yes         |
    | decoder.layers.5.ln3 (LayerNorm)                      | [--]           | 1,024      | Yes         |
    | output_layer (Linear)                                 | [--]           | 1,936,575  | Yes         |
    +-------------------------------------------------------+----------------+------------+-------------+

    ============================================================
    Total params: 186.37M (186,372,287)
    Trainable params: 186.37M (186,372,287)
    Non-trainable params: 0 (0)
    Model size: 710.95MB (FP32)
    ============================================================
```

可以看到模型的参数量大约为 1.86 亿, 模型大小大约为 710MB. 

### 模型训练及评估

该部分的代码与常规深度学习训练过程类似, 故不再赘述, 需要注意的细节可见注释, 代码可见 `main.py` 文件. 

训练完成后, 我们可以查看一下训练的损失和准确率的变化曲线. 其中, 训练集和验证集上的 Loss 以及 Accuracy 的变化曲线如下图所示: 

![alt text](figs/accuracy_curve.png)

![alt text](figs/loss_curve.png)

从图中可以发现, 虽然随着训练 epoch 的增加, 训练集的 Loss 持续下降, Accuracy 持续上升, 但是在第 10 个 epoch 之后, 验证集的 Loss 就开始上升, Accuracy 开始下降, 这说明模型出现了过拟合的现象. 因此, 在该超参数设置下, 模型的最佳效果出现在第 10 个 epoch, 此时验证集的 Loss 最小, Accuracy 最大. 我们可以将该 epoch 的模型保存下来, 作为最终的模型. 

5. 模型评估

    在模型评估时, 我们使用上文提到的训练了 10 个 epoch 的模型, 并使用 BLEU 分数来评估模型的翻译效果. BLEU 分数是一个常用的机器翻译评估指标, 主要用于衡量机器翻译结果与参考翻译之间的相似度. BLEU 分数越高, 表示翻译结果越好. 
    在这里, 我们使用 `nltk.translate.bleu_score` 库中的 `corpus_bleu` 函数来计算 BLEU 分数. 代码实现如下: 

    ```python
    from nltk.translate.bleu_score import corpus_bleu

    # ......
    elif period == "eval":
        model_path = os.path.join(output_dir, "final_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found.")
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        with torch.no_grad():
            en_origin = []
            cn_standard = []
            cn_output = []
            for batch in tqdm(test_loader):
                x = torch.LongTensor(batch["source"]).to(device)
                y = torch.LongTensor(batch["target"]).to(device)
                batch_size = x.shape[0]
                max_len = y.shape[1]
                y_output = torch.full(
                    (batch_size, max_len), PAD_ID, dtype=torch.long, device=device
                )
                y_output[:, 0] = BOS_ID
                for cur_idx in range(1, max_len):
                    decoder_input = y_output[:, :cur_idx]
                    output = model(x, decoder_input)
                    next_tokens = torch.argmax(output[:, -1, :], dim=-1)
                    y_output[:, cur_idx] = next_tokens
                for j in range(batch_size):
                    en_origin.append(convert_to_text(x[j], int2en))
                    cn_standard.append(convert_to_text(y[j], int2cn))
                    cn_output.append(convert_to_text(y_output[j], int2cn))
        references = [[ref.split()] for ref in cn_standard]
        hypotheses = [hyp.split() for hyp in cn_output]
        bleu_score = corpus_bleu(references, hypotheses)
        print(f"BLEU Score: {bleu_score:.4f}")
    ```

### 案例分析

此外, 我们还可以手动检验模型的翻译效果, 代码实现如下: 

```python
print("-" * 50)
for i in range(3):
        print(f"Original: {en_origin[i]}")
        print(f"Standard: {cn_standard[i]}")
        print(f"Translated: {cn_output[i]}")
        print("-" * 50)
```

最终的输出结果如下: 

```txt
BLEU Score: 0.2766
--------------------------------------------------
Original: do you still want to talk to me ?
Standard Answer: 你 还 想 跟 我 谈 吗 ？
Translated: 你 还 想 跟 我 说 吗 ？
--------------------------------------------------
Original: i will never for ce you to marry him .
Standard Answer: 我永远 不会 逼 你 跟 他 结婚 . 
Translated: 我 不会 忘记 你 和 他 结婚 了 . 
--------------------------------------------------
Original: i 'm going to go tell tom .
Standard Answer: 我要 告诉 汤姆 . 
Translated: 我要 告诉 汤姆 . 
--------------------------------------------------
```

BLEU 分数为 0.2322, 说明模型的翻译效果还不错. 同时, 我们可以看到, 手动输出的几个示例中, 模型的翻译效果也较为理想. 虽然有些地方翻译得不是很正确, 例如将 "force" 翻译成了 "忘记", 但整体上还是符合逻辑且较为准确的. 

### 消融实验

在消融实验部分, 我主要从模型结构方面进行了消融实验, 主要探究了不同的注意力头数、不同的编码器和解码器层数和是否使用位置编码对模型性能的影响. 

#### 注意力头数

在 Transformer 模型中, 注意力头数是一个重要的超参数, 它决定了模型在处理序列数据时, 是否能够捕捉到不同位置之间的依赖关系. 当头数较多时, 模型能够并行地在子空间中捕获不同位置之间的依赖关系, 从而提高模型的表达能力, 但同时也可能因影响子空间的大小而降低模型的表达能力. 当头数过少时, 模型可能无法捕捉到足够的信息, 从而影响模型的性能. 因此, 我们需要探究注意力头数对模型性能的影响. 

在本次实验中, 我设置了 5 个不同的注意力头数, 分别为 1, 2, 4, 8, 16, 并计算对应的 BLEU 分数. 实验结果如下: 

| 注意力头数 | BLEU 分数 |
|------------|-----------|
| 1          | 0.2307    |
| 2          | 0.2371    |
| 4          | 0.2438    |
| 8          | **0.2766**    |
| 16         | 0.2408    |

从实验结果可以看出, 当注意力头数为 8 时, 模型的 BLEU 分数最高, 为 0.2766. 当注意力头数为 1 时, 模型的 BLEU 分数最低, 为 0.2307. 注意力头数为 16 时, 模型的 BLEU 分数为 0.2408, 略低于注意力头数为 8 时的 BLEU 分数. 由此也可以看出, 注意力头数并非越多越好, 或越少越好, 而是需要根据具体任务和数据集进行调整. 

#### 编码器和解码器层数

在 Transformer 模型中, 编码器和解码器层数也是十分重要的超参数. 模型的层数直接影响深度和大小, 从而影响模型的表达能力. 

在本次实验中, 我也设置了 5 个不同的编码器和解码器层数, 分别为 1, 2, 4, 6, 8, 10, 并计算对应的 BLEU 分数. 实验结果如下: 

| 编码器和解码器层数 | BLEU 分数 |
|------------------|-----------|
| 1                | 0.2134    |
| 2                | 0.2329    |
| 4                | 0.2428    |
| 6                | **0.2766**    |
| 8                | 0.2594    |
| 10               | 0.2293    |

从实验结果可以看出, 当层数为 6 时, BLEU 分数最高. 而层数为 10 时, 效果甚至不如层数为 2 时的效果. 这可能是因为层数过多时, 模型可能出现过拟合的现象, 从而影响模型的性能. 

### 位置编码

位置编码对于 Transformer 模型来说, 是一个十分重要的模块. 位置编码的目的是为了将序列中的位置信息编码成向量, 从而让模型能够捕捉到序列中的顺序信息. 若没有位置编码, 模型在并行化处理时, 会丢失序列的顺序信息, 从而影响模型的性能. 因此, 我们需要探究位置编码对模型性能的影响. 

| 位置编码 | BLEU 分数 |
|------|-----------|
| ❌ | 0.2175    |
| ✅ | **0.2766**    |

从实验结果可以看出, 当使用位置编码时, 模型的 BLEU 分数高于不使用位置编码时的 BLEU 分数, 且提升了 0.0591. 由此也可以看出, 位置编码对于 Transformer 模型来说, 是一个十分重要的模块. 





## 💙 项目心得

通过本次实验，我对 Transformer 模型的结构与原理有了更加全面、深入的认识。首先，我详细剖析了多头自注意力机制（Multi‐Head Self‐Attention）、前馈网络（Feed‐Forward Network）以及残差连接与层归一化（Residual Connection & Layer Normalization）三大核心模块的内部运作原理，并通过绘制模型结构图加深记忆。

在实现环节，我使用 PyTorch 从零构建了一个简化版的 Transformer 编码器–解码器架构，包括词嵌入（Embedding）、位置编码（Positional Encoding）和掩码机制（Masking）的完整流水线。通过调试和单元测试，我掌握了如何在代码层面灵活地控制注意力权重的计算、梯度反向传播与参数更新。

为了验证模型在机器翻译任务上的效果，我采用了标准的 BLEU 分数作为定量评估指标，并在中英对照语料集上进行了多轮实验。实验结果显示，当训练轮次达到 10 轮以上时，模型就出现了过拟合现象，验证集的 BLEU 分数开始下降。因此，我选择在第 10 轮训练结束后保存模型，并使用该模型进行翻译任务。

在手动检验环节，我随机选取了 3 个测试样本，并使用训练好的模型进行翻译。通过对比标准答案与模型翻译结果，我发现尽管存在一些细节上的不足，但整体翻译效果还是较为令人满意的。

消融实验部分, 我系统地探究了注意力头数、编码器和解码器层数和位置编码对模型性能的影响. 其中, 位置编码对 Transformer 模型的性能极大, 而注意力头数和编码器和解码器层数则需要根据具体的任务和数据集的大小进行调整, 选择最合适的参数. 

通过这次实践，我不仅加深了对前沿模型的理论理解，也在工程实现与实验评估方面积累了宝贵经验，为后续更复杂的自然语言处理项目打下了坚实基础。


## 📺 演示视频

[演示视频](https://box.nju.edu.cn/f/e0f8854b0b0a47fbac4c/) 中包含了模型的训练和评估过程.

## 📜 参考资料

[1] [Attention Is All You Need. NeurIPS 2017](https://arxiv.org/abs/1706.03762)

[2] [Yi-Fan Zhou's Blog](https://zhuanlan.zhihu.com/p/581334630)

[3] [机器翻译评价指标BLEU介绍](https://blog.csdn.net/g11d111/article/details/100103208)

[4] [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

[5] [Learning Deep Transformer Models for Machine Translation. ACL 2019](https://arxiv.org/abs/1906.01787)

[6] [《神经网络与深度学习》 邱锡鹏](https://nndl.github.io/)