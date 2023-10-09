# coding: UTF-8
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, test_iter):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    total_batch = 0  # 记录进行到多少batch
    best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (x, labels) in enumerate(train_iter):
            # print (x[0].shape)
            x = x.to(config.device)
            labels = labels.to(config.device)
            outputs = model(x)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                test_acc, test_loss = evaluate(config, model, test_iter)
                if test_loss < best_loss:
                    best_loss = test_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = 'yes'
                    last_improve = total_batch
                else:
                    improve = 'no'
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, Improve: {5}'
                print(msg.format(total_batch, loss.item(), train_acc, test_loss, test_acc, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过一定的batch没下降，结束训练
                print(
                    f"No optimization for a long time: {config.require_improvement} batch no improve, auto-stopping...")
                flag = True
                break
        if flag:
            break


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)


def predict(config, model, predict_iter):
    # predict
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    domain_all = np.array([], dtype=str)
    predict_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, x, labels in predict_iter:
            x = x.to(config.device)
            outputs = model(x)
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            predict_all = np.append(predict_all, predict)
            domain_all = np.append(domain_all, texts)
    print(predict_all)
    print(domain_all)
    df = pd.DataFrame()
    df['domain'] = domain_all
    df['label'] = predict_all
    df.to_csv('predict.csv', index=False, header=False)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            texts = texts.to(config.device)
            labels = labels.to(config.device)
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)
