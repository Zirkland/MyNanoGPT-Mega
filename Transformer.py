from DataLoader import hyperparameters
import math
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len):
        super().__init__()
        self.pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(math.log(10000.0) / embedding_dim))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = hyperparameters['num_heads']
        self.embedding_dim = hyperparameters['embedding_dim']
        self.head_dim = self.embedding_dim // self.num_heads
        self.batch_size = hyperparameters['batch_size']
        assert self.head_dim * self.num_heads == self.embedding_dim, 'embedding_dim必须是num_heads的整数倍,否则错误定义的超参会导致不可预知的错误'

        # 定义Q,K,V的线性层
        self.q = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.k = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.v = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.out = nn.Linear(self.embedding_dim, self.embedding_dim)

        # 生成一个下三角矩阵
        self.pre_mask = torch.tril(torch.ones(hyperparameters['block_size'], hyperparameters['block_size']))
        # block_size是一个block的长度，读取与输出顺着这个方向进行，所以block_size即时间维度
        # 故这是一个时间维度的下三角矩阵，顺着时间维度进行掩码，就不会看到未来的信息，从而保证了模型的自回归性质
        self.mask = self.pre_mask.masked_fill(self.pre_mask == 0, float('-inf'))

    def forward(self, x):
        # 确保mask在与输入张量x相同的设备上
        self.mask = self.mask.to(x.device)

        # x的shape是[batch_size, block_size, embedding_dim]
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # 将Q,K,V进行分割成num_heads个头
        q = q.view(self.batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(self.batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(self.batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # permute(0,2,1,3)是将维度进行转换，比如原来的维度是[batch_size, block_size, num_heads, head_dim]
        # 转换后就是[batch_size, num_heads, block_size, head_dim]
        # 原本的查询（Q）、键（K）和值（V）向量是三维的，其中中间的维度是 embedding_dim。
        # 在多头注意力机制中，这些向量会被分割为多个头，每个头的维度为 (num_heads, head_dim)，从而得到四维的注意力头 Q、K 和 V

        # 计算Q,K的点积，然后除以根号head_dim，这是为了防止点积过大，导致梯度过小
        k_t = k.permute(0, 1, 3, 2)
        attention = torch.matmul(q, k_t) / math.sqrt(self.head_dim)
        attention = attention + self.mask  # 加上掩码,这样就不会看到未来的信息
        attention = torch.softmax(attention, dim=-1)
        # 计算完了注意力分数张量,形状是[batch_size, num_heads, block_size, block_size]
        # V的形状是[batch_size, num_heads, block_size, head_dim]，attention.shape[-1]是block_size,与V的倒数第二维相同
        # 所以可以直接点乘，得到输出
        output = torch.matmul(attention, v)
        output = output.permute(0, 2, 1, 3).contiguous().view(self.batch_size, -1, self.embedding_dim)
        # 将输出的形状转换成[batch_size, block_size, embedding_dim]
        output = self.out(output)
        # 最后再过一个线性层
        return output


# 这边定义了一个LayerNorm，其实就是Transformer中的ADD&Norm，其作用是防止梯度消失或爆炸
class LayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hyperparameters['embedding_dim']))  # 权重矩阵是一个可学习的参数
        self.bias = nn.Parameter(torch.zeros(hyperparameters['embedding_dim']))  # 偏置矩阵是一个可学习的参数
        self.eps = 1e-6  # 定义一个很小的数，防止产生除0错误

    def forward(self, x):
        # 在层归一化（Layer Normalization）中，“层”指的是输入的特征维度（即embedding_dim）。
        # 具体来说，层归一化会对每个样本的所有特征进行归一化处理。
        mean = x.mean(-1, keepdim=True)  # 得到均值
        std = x.std(-1, keepdim=True)  # 得到标准差
        x_prime = (x - mean) / (std + self.eps)
        return x_prime * self.weight + self.bias

# 定义一个Transformer中的前馈神经网络,实际上就是一个全连接层，简单地说就是两个线性层中间加一个激活函数
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_dim = hyperparameters['embedding_dim']
        self.ffn_dim = hyperparameters['ffn_dim']
        self.linear1 = nn.Linear(self.embedding_dim, self.ffn_dim)  # x 计算dot W1+b-->X, Y=max(0,X)在下面的forward中实现
        self.linear2 = nn.Linear(self.ffn_dim, self.embedding_dim)  # Y 计算dotW2+b-->output

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)  # 计算Y=max(0,X)
        x = self.linear2(x)
        return x

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(hyperparameters['vocab_size'], hyperparameters['embedding_dim'])  # 词嵌入
        self.positional_encoding = PositionalEncoding(hyperparameters['embedding_dim'], hyperparameters['max_len'])
        self.MultiHeadAttention = MaskedMultiHeadAttention()
        self.LayerNorm1 = LayerNorm()
        self.FeedForward = FeedForward()
        self.LayerNorm2 = LayerNorm()
        self.dropout = nn.Dropout(0.1)  # Add Dropout layer
        self.linear = nn.Linear(hyperparameters['embedding_dim'], hyperparameters['vocab_size'])
        self.softmax = nn.Softmax(dim=-1)
        # 最后的输出层,将embedding_dim映射到vocab_size
        # 得到一个概率分布的输出，形状是[batch_size, block_size, vocab_size]

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.MultiHeadAttention(x)
        x = self.LayerNorm1(x)
        x = self.FeedForward(x)
        x = self.LayerNorm2(x)
        x = self.dropout(x)  # Apply Dropout
        x = self.linear(x)
        x = self.softmax(x)
        return x
