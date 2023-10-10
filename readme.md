# DGA域名检测
## 数据
./train 训练样本，黑白各100万

./test 测试样本，无标签
## 数据预处理
字符级别的word2vec
## 模型
双向LSTM
## 结果
截取训练过程的两个epoch如下：

Epoch [117/200]

Iter:  26000,  Train Loss:  0.33,  Train Acc: 98.49%,  Val Loss:  0.33,  Val Acc: 97.85%, Improve: no

Iter:  26100,  Train Loss:  0.33,  Train Acc: 98.69%,  Val Loss:  0.33,  Val Acc: 97.76%, Improve: no

Iter:  26200,  Train Loss:  0.33,  Train Acc: 98.53%,  Val Loss:  0.33,  Val Acc: 97.88%, Improve: no

Epoch [118/200]

Iter:  26300,  Train Loss:  0.33,  Train Acc: 98.57%,  Val Loss:  0.33,  Val Acc: 97.90%, Improve: yes

Iter:  26400,  Train Loss:  0.33,  Train Acc: 98.64%,  Val Loss:  0.33,  Val Acc: 97.85%, Improve: no
