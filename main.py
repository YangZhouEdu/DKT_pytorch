import torch
import argparse
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Model.DKT import  DKT
import torch.nn as nn
from data_loader import DataReader
from Dataset.DKTDataset import DKTDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
"""
author:'SUN HAO'
"""
def train_epoch(model, train_iterator, optim, criterion,parsers, device="cpu"):
    model.train()

    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    tbar = tqdm(train_iterator)
    for item in tbar:
        x = item[0].to(device).float()
        target_id = item[1].to(device).long()
        label = item[2].to(device).float()
        target_mask = (target_id != 0)
        optim.zero_grad()
        output = model(x)
        output = torch.gather(output, 2, target_id.unsqueeze(2) )
        output = torch.masked_select(output.squeeze(2), target_mask)
        label = torch.masked_select(label, target_mask)

        loss = criterion(output, label)
        loss.backward()
        optim.step()
        train_loss.append(loss.item())
        pred = (torch.sigmoid(output) >= 0.5).long()

        num_corrects += (pred == label).sum().item()
        num_total += len(label)

        labels.extend(label.view(-1).data.cpu().numpy())
        outs.extend(output.view(-1).data.cpu().numpy())

        tbar.set_description('loss - {:.4f}'.format(loss))

    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)

    loss = np.average(train_loss)

    return loss, acc, auc
def val_epoch(model, val_iterator, criterion,parsers, device="cpu"):
    model.eval()

    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    tbar = tqdm(val_iterator)
    for item in tbar:
        x = item[0].to(device).float()
        target_id = item[1].to(device).long()
        label = item[2].to(device).float()
        target_mask = (target_id != 0)

        with torch.no_grad():
            output = model(x)

        output = torch.gather(output, 2, target_id.unsqueeze(2))
        output = torch.masked_select(output.squeeze(2), target_mask)
        label = torch.masked_select(label, target_mask)

        loss = criterion(output, label)
        train_loss.append(loss.item())

        pred = (torch.sigmoid(output) >= 0.5).long()

        num_corrects += (pred == label).sum().item()
        num_total += len(label)

        labels.extend(label.view(-1).data.cpu().numpy())
        outs.extend(output.view(-1).data.cpu().numpy())

        tbar.set_description('loss - {:.4f}'.format(loss))

    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)
    loss = np.average(train_loss)

    return loss, acc, auc


# 解析传入的参数
parser = argparse.ArgumentParser(description='myDemo')
parser.add_argument('--batch-size',type=int,default=64,metavar='N',help='number of batch size to train (defauly 64 )')
parser.add_argument('--epochs',type=int,default=30,metavar='N',help='number of epochs to train (defauly 10 )')
parser.add_argument('--lr',type=float,default=0.01,help='number of learning rate')
parser.add_argument('--data_dir', type=str, default='./data/',help="the data directory, default as './data")
parser.add_argument('--hidden_size',type=int,default=200,help='the number of the hidden-size')
parser.add_argument('--max_step',type=int,default=100,help='the number of max step')
parser.add_argument('--num_layers',type=int,default=1,help='the number of layers')
parser.add_argument('--separate_char',type=str,default=',',help='分隔符')
parser.add_argument('--min_step',type=int,default=10,help='the number of min step')
# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    dataset = 'ASSISTments2009'  #  ASSISTments2009 / ASSISTment2015 /  synthetic / statics2011 / junyiacademy / EDNet

    model = 'DKT'  # DKT /
    if dataset == 'ASSISTments2009':
        parser.add_argument('--dataset', type=str, default='ASSISTments2009', help='which dataset to train')
        parser.add_argument('--train_file', type=str, default='train_set.csv',
                            help="train data file, default as 'train_set.csv'.")
        parser.add_argument('--test_file', type=str, default='test_set.csv',
                            help="train data file, default as 'test_set.csv'.")
        parser.add_argument('--save_dir_prefix', type=str, default='./ASSISTments2009/',
                            help="train data file, default as './ASSISTments2009/'.")
        parser.add_argument('--n_question', type=int, default=123, help='the number of unique questions in the dataset')

    elif dataset == 'ASSISTment2015':
        parser.add_argument('--dataset', type=str, default='ASSISTment2015', help='which dataset to train')
        parser.add_argument('--train_file', type=str, default='train_set.csv',
                            help="train data file, default as 'train_set.csv'.")
        parser.add_argument('--test_file', type=str, default='test_set.csv',
                            help="train data file, default as 'test_set.csv'.")
        parser.add_argument('--save_dir_prefix', type=str, default='./ASSISTment2015/',
                            help="train data file, default as './ASSISTment2015/'.")
        parser.add_argument('--n_question', type=int, default=100, help='the number of unique questions in the dataset')

    elif dataset == 'synthetic':
        parser.add_argument('--dataset', type=str, default='synthetic', help='which dataset to train')
        parser.add_argument('--train_file', type=str, default='naive_c5_q50_s4000_v0train_set.csv',
                            help="train data file, default as 'naive_c5_q50_s4000_v0train_set.csv'.")
        parser.add_argument('--test_file', type=str, default='naive_c5_q50_s4000_v0test_set.csv',
                            help="train data file, default as 'naive_c5_q50_s4000_v0test_set.csv'.")
        parser.add_argument('--save_dir_prefix', type=str, default='./synthetic/',
                            help="train data file, default as './synthetic/'.")
        parser.add_argument('--n_question', type=int, default=1224, help='the number of unique questions in the dataset')

    elif dataset == 'statics2011':
        parser.add_argument('--dataset', type=str, default='statics2011', help='which dataset to train')
        parser.add_argument('--train_file', type=str, default='train_set.csv',
                            help="train data file, default as 'train_set.csv'.")
        parser.add_argument('--test_file', type=str, default='test_set.csv',
                            help="train data file, default as 'test_set.csv'.")
        parser.add_argument('--save_dir_prefix', type=str, default='./statics2011/',
                            help="train data file, default as './statics2011/'.")
        parser.add_argument('--n_question', type=int, default=50, help='the number of unique questions in the dataset')


    elif dataset =='junyiacademy':
        parser.add_argument('--dataset', type=str, default='junyiacademy', help='which dataset to train')
        parser.add_argument('--train_file', type=str, default='problem_train_set.csv',
                            help="train data file, default as 'train_set.csv'.")
        parser.add_argument('--test_file', type=str, default='problem_test_set.csv',
                            help="train data file, default as 'test_set.csv'.")
        parser.add_argument('--save_dir_prefix', type=str, default='./junyiacademy/',
                            help="train data file, default as './junyiacademy/'.")
        # parser.add_argument('--n_question', type=int, default=1326, help='the number of unique questions in the dataset')
        parser.add_argument('--n_question', type=int, default=25784, help='the number of unique questions in the dataset')

    elif dataset == 'EDNet':
        parser.add_argument('--dataset', type=str, default='EDNet', help='which dataset to train')
        parser.add_argument('--train_file', type=str, default='train_set.csv',
                            help="train data file, default as 'train_set.csv'.")
        parser.add_argument('--test_file', type=str, default='test_set.csv',
                            help="train data file, default as 'test_set.csv'.")
        parser.add_argument('--save_dir_prefix', type=str, default='./EDNet/',
                            help="train data file, default as './EDNet/'.")
        parser.add_argument('--n_question', type=int, default=13168, help='the number of unique questions in the dataset')




    # 解析参数
    parsers = parser.parse_args()

    print("parser:",parsers)
    # train 路径 和 test 路径
    print(f'loading Dataset  {parsers.dataset}...')
    train_path = parsers.data_dir + parsers.dataset + '/' + parsers.train_file
    test_path = parsers.data_dir + parsers.dataset + '/' + parsers.test_file
    train =   DataReader(path=train_path, separate_char=parsers.separate_char)
    train_set =  train.load_data()
    test = DataReader(path=test_path,separate_char=parsers.separate_char)
    test_dataset = test.load_data()

    train_set = pd.DataFrame(train_set,columns=['user_id','skill_id','correct']).set_index('user_id')
    test_dataset = pd.DataFrame(test_dataset, columns=['user_id', 'skill_id', 'correct']).set_index('user_id')


    train_dataset = DKTDataset(group=train_set,n_skill=parsers.n_question,max_seq=parsers.max_step,min_step=parsers.min_step)
    # print(train_dataset.__dict__)
    # 使用固定缓冲区 加快Tensor复制的速度，并且可以使用异步的方式进行复制

    dataloader_kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}

    test_dataset = DKTDataset(test_dataset, n_skill=parsers.n_question, max_seq=parsers.max_step, min_step=parsers.min_step)
    # 如果运行内存不够建议减低num_workers的值
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=5, **dataloader_kwargs)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=5, **dataloader_kwargs)
    # 使用GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = DKT(input_dim=parsers.n_question*2,hidden_dim=parsers.hidden_size,layer_dim=parsers.num_layers,output_dim = parsers.n_question)
    print(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=parsers.lr)
    # nn.BCELoss() 与 nn.BCEWithLogitsLoss() 区别在于 前者进行了sigmoid计算
    criterion = nn.BCEWithLogitsLoss()

    criterion.to(device)


    for epoch in range(parsers.epochs):
        train_loss, train_acc, train_auc = train_epoch(model, train_dataloader, optimizer, criterion,parsers, device)
        print(
            "epoch - {} train_loss - {:.2f} acc - {:.3f} auc - {:.3f}".format(epoch, train_loss, train_acc, train_auc))

        val_loss, avl_acc, val_auc = val_epoch(model, test_dataloader, criterion,parsers, device)
        print("epoch - {} test_loss - {:.2f} acc - {:.3f} auc - {:.3f}".format(epoch, val_loss, avl_acc, val_auc))
