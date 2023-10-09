import torch
import torch.nn as nn


class Config(object):
    """配置参数"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.log_path = 'log'
        self.num_works = 4
        self.dropout = 0.4  # 随机失活
        self.save_path = 'BiLSTM.ckpt'
        self.require_improvement = 100000  # 若超过 xxxx batch效果还没提升，则提前结束训练
        self.num_classes = 2
        self.num_epochs = 500  # epoch数
        self.batch_size = 10000  # mini-batch大小
        self.pad_size = 74  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率
        self.embed = 300  # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 128  # lstm隐藏层
        self.num_layers = 2  # lstm层数
        self.mid_fc_size = 64  # 最后的过渡全连接层神经元数量
        self.n_vocab = 0  # 词表大小，在运行时赋值


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=0)

        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.mid_fc_size),
            nn.Dropout(p=config.dropout),
            nn.SELU(),
            nn.Linear(config.mid_fc_size, config.num_classes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # [batch_size, seq_len, embeding]=[128, 32, 300]
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        out = self.softmax(out)
        return out
