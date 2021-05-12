import torch
import torch.nn as nn
from torch.autograd import Variable
# 导入torch.nn模块。实际上，“nn”是neural networks（神经网络）的缩写
#  nn.RNN的参数设置
#  input_size   输入x的特征大小(以mnist图像为例，特征大小为28*28 = 784)
#  hidden_size   隐藏层h的特征大小 即隐藏层节点的个数
#  num_layers    循环层的数量（RNN中重复的部分） 层数
#  nonlinearity   激活函数 默认为tanh，可以设置为relu
#  bias   是否设置偏置，默认为True
#  batch_first   默认为false, 设置为True之后，输入输出为(batch_size, seq_len, input_size)
#  dropout   默认为0
#  bidirectional   默认为False，True设置为RNN为双向


class DKT(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(DKT, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True,nonlinearity='tanh')
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)


    def forward(self, x):
        device = x.device
        # torch.zeros 返回一个形状为为size,类型为torch.dtype，里面的每一个值都是0的tensor
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(device)
        out,hn = self.rnn(x, h0)
        res = self.fc(out)
        return res


# 输出形状
# DKT(
#   (rnn): RNN(246, 100, dropout=0.02)
#   (dense): Linear(in_features=100, out_features=123, bias=True)
#   (sig): Sigmoid()
#  )