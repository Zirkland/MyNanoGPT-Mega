import glob
import json
import re
from symbol import parameters

import torch


# 把json文件中的古文提取出来，然后把古文转换成对应的token
pattern = re.compile(r'[\u4e00-\u9fa5，。《》]')

json_files = glob.glob('.\\AllAncientWorks\\*.json')
paragraphs_list = []

for file_name in json_files:
    with open(file_name, 'r', encoding='utf-8', errors='ignore') as f:
        try:
            text = json.load(f)
            # 只需要paragraphs字段(故文内容)
            for item in text:
                filtered_paragraphs = [''.join(pattern.findall(paragraph)) for paragraph in item.get('paragraphs', [])]
                paragraphs_list.extend(filtered_paragraphs)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file {file_name}: {e}")

# 把上面得到的语料表转化为一个string
combined_text = ''.join(paragraphs_list)

# 生成vocab字典与反字典
vocab = sorted(set(combined_text))
stoi = {char: i for i, char in enumerate(vocab)}
itos = {i: char for i, char in enumerate(vocab)}

#定义编码与解码函数
encode = lambda s: [stoi[c] for c in s]
decode = lambda x: ''.join([itos[i] for i in x])

# 把古文转换成token
data = torch.tensor(encode(combined_text), dtype=torch.long)
# print(data.tolist())
n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]

#超参字典，方便其他地方调用
hyperparameters={
'device':'cuda',#启动我的4080laptop！！！
'block_size':16,#每次输入的长度
'batch_size':256,#每次输入的数量(一批多少个)
'vocab_size':len(vocab),
'embedding_dim':512,#嵌入维度,在embedding与MaskedMultiHeadAttention中被调用
'max_len':512,#最大长度,在PositionalEncoding中被调用,这里为了方便被设置为embedding_dim相同,但是可以自己定义为别的值
'num_heads':8,#多头注意力的头数,在MaskedMultiHeadAttention中被调用
'ffn_dim':2048,#前馈神经网络的维度,在FeedForward中被调用
'learning_rate':1e-4,#学习率
'max_iters':1000000,#最大迭代次数，就是总训练轮数
'eval_interval':400, #每隔多少次迭代进行一次验证
'log_interval':250, #每隔多少次迭代输出一次日志
}

#定义获取batch的函数，返回x,y，x是输入，y是输出，x和y的shape都是[batch_size,block_size]，y是x的后一个字符，这样就可以用x预测y，即语言模型，这里的x和y都是tensor
def get_batch(split):
    data = train_data if split == 'train' else val_data
    start_idx = torch.randint(0, len(data) - hyperparameters['block_size'], (hyperparameters['batch_size'],))  # Randomly generate a set of start positions
    x = torch.stack([data[i:i + hyperparameters['block_size']] for i in start_idx])
    y = torch.stack([data[i + 1:i + hyperparameters['block_size'] + 1] for i in start_idx])
    return x.to(hyperparameters['device']), y.to(hyperparameters['device'])