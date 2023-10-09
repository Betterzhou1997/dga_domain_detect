import os.path
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class DGADataset(Dataset):  # 继承Dataset类
    def __init__(self, is_train: bool, max_len=74, is_predict=False):
        self.is_train = is_train
        self.is_predict = is_predict
        if self.is_predict:
            data = pd.read_csv('test/test_domains.csv')
        else:
            data = pd.concat([pd.read_csv('train/black.csv'), pd.read_csv('train/white.csv')])
        x = data['domain'].values
        y = data['label'].values
        x_features = []
        if not os.path.exists('char_dict.pickle'):
            print("char_dict not exist, creating...")
            char_dict = {}
            for domain in x:
                for char in domain:
                    if char in char_dict:
                        continue
                    else:
                        char_dict[char] = len(char_dict) + 1
            char_dict['UNK'] = len(char_dict) + 1

            # 将字典保存到磁盘中
            with open('char_dict.pickle', 'wb') as f:
                pickle.dump(char_dict, f)
        else:
            # 从磁盘中加载字典
            print("char_dict already exist, loading...")
            with open('char_dict.pickle', 'rb') as f:
                char_dict = pickle.load(f)
        print(char_dict)
        for domain in x:
            # print(domain)
            # 转为数字
            idx = domain.find(':')
            if idx != -1:
                domain = domain[:idx]

            each_feature = [char_dict[i] if i in char_dict else char_dict['UNK'] for i in domain]
            # 过长删掉
            if len(each_feature) >= max_len:
                each_feature = each_feature[:max_len]
            # 过短补0
            else:
                each_feature.extend([0] * (max_len - len(each_feature)))
            assert len(each_feature) == max_len
            x_features.append(each_feature)

        assert len(x_features) == len(y)
        if self.is_predict:
            print("loading predict data")
            self.text = x
            self.x = x_features
            self.y = y
        else:
            x_train, x_test, y_train, y_test = train_test_split(x_features, y, test_size=0.2, random_state=42, stratify=y)
            if is_train:
                self.x = x_train
                self.y = y_train
            else:
                self.x = x_test
                self.y = y_test

    def __getitem__(self, index):
        if self.is_predict:
            text = self.text[index]
            x = torch.tensor(self.x[index])
            y = torch.tensor(self.y[index])
            return text, x, y

        else:
            x = torch.tensor(self.x[index])
            y = torch.tensor(self.y[index])
            return x, y

    def __len__(self):
        return len(self.y)


if __name__ == '__main__':
    train_dataset = DGADataset(is_train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True,
                              num_workers=2, pin_memory=True)
    # test_dataset = DGADataset(is_train=False)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=True,
    #                          num_workers=2, pin_memory=True)
    predict_dataset = DGADataset(is_train=False, is_predict=True)
    predict_loader = DataLoader(dataset=predict_dataset, batch_size=100, shuffle=True,
                                num_workers=2, pin_memory=True)
    with open('char_dict.pickle', 'rb') as f:
        char_dict = pickle.load(f)

    # for texts, x, y in predict_loader:
    #     print(texts)
    #     for i in range(len(texts)):
    #         domain = texts[i]
    #         each_feature = [char_dict[i] if i in char_dict else char_dict['UNK'] for i in domain]
    #         print(each_feature)
    #         print(x[i])
    #     break
    for x, y in train_loader:
        print(len(y), sum(y))

