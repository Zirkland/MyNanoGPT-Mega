from DataLoader import get_batch
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self,embedding_dim,max_len):
        super().__init__()
        self.pe=nn.Parameter(torch.zeros(1,max_len,embedding_dim))#若不想让pe被训练，禁用这条
        # pe = torch.zeros(max_len, embedding_dim)
        # 若不想让pe被训练，启用这条
        position=torch.arange(0,max_len).unsqueeze(1).float()
        div_term=torch.exp(torch.arange(0,embedding_dim,2).float()*-(math.log(10000.0)/embedding_dim))
        self.pe[:,0::2]=torch.sin(position*div_term)
        self.pe[:,1::2]=torch.cos(position*div_term)
        # pe=self.pe.unsqueeze(0)
        # 若不想让pe被训练，启用这条
        # self.register_buffer('pe',pe)
        #若不想让pe被训练，启用这条
    def forward(self,x):
        return x+self.pe[:,:x.size(1)]
#继承nn.Module得到类MyModule
class MyModel(nn.Module):
    def __init__(self,vocab_size,embedding_dim):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_dim)#词嵌入
        self.PositionalEncoding=PositionalEncoding(embedding_dim,)
