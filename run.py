import pickle

import torch
from torch.utils.data import DataLoader
from model import Config, Model
from data_process import DGADataset

from train_eval import train, predict

if __name__ == '__main__':
    print(torch.cuda.is_available())
    # torch.manual_seed(1)
    # torch.cuda.manual_seed_all(1)
    # torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    config = Config()

    for attr in dir(config):
        if not callable(getattr(config, attr)) and not attr.startswith("__"):
            print(attr, getattr(config, attr))

    train_dataset = DGADataset(is_train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_works, pin_memory=True)
    test_dataset = DGADataset(is_train=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_works, pin_memory=True)

    predict_dataset = DGADataset(is_train=False, is_predict=True)
    predict_loader = DataLoader(dataset=predict_dataset, batch_size=config.batch_size, shuffle=True,
                                num_workers=config.num_works, pin_memory=True)
    with open('char_dict.pickle', 'rb') as f:
        char_dict = pickle.load(f)
    config.n_vocab = len(char_dict)
    model = Model(config).to(config.device)
    print(model)
    train(config, model, train_loader, test_loader)
    predict(config, model, predict_loader)