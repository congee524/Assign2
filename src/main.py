import argparse
import os
import random

import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import FeatureDataset
from model import Classifier
from utils import SmoothedValue, cal_acc


def get_args():
    parser = argparse.ArgumentParser('AML Assign2', add_help=False)

    parser.add_argument('--mode',
                        choices=['train', 'test'],
                        default='train',
                        help='running mode')

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default='0.1', type=float)

    parser.add_argument('--anno', default='', type=str)
    parser.add_argument('--validation', action='store_true', default=False)

    parser.add_argument('--work_dir', default='work_dir/', type=str)

    parser.add_argument('--num_classes', default=6, type=int)
    parser.add_argument('--in_channels', default=3072, type=int)

    parser.add_argument('--load_from',
                        default='',
                        type=str,
                        help='load model parameters')

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    # dataset
    anno_file = args.anno
    num_data = len(pd.read_csv(anno_file, delimiter=','))

    if args.mode == 'train':
        if args.validation:
            val_data_list = random.sample(range(num_data), num_data // 6)
            train_data_list = [
                train_idx for train_idx in range(num_data)
                if train_idx not in val_data_list
            ]
            train_dataset = FeatureDataset(anno_file,
                                           train_data_list,
                                           mode='train')
            val_dataset = FeatureDataset(anno_file, val_data_list, mode='val')
        else:
            train_data_list = list(range(num_data))
            train_dataset = FeatureDataset(anno_file,
                                           train_data_list,
                                           mode='train')

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      drop_last=True)
        if args.validation:
            val_dataloader = DataLoader(val_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=False)
    elif args.mode == 'test':
        test_data_list = list(range(num_data))
        test_dataset = FeatureDataset(anno_file, test_data_list, mode='test')
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = Classifier(args.num_classes, args.in_channels).cuda()
    if args.mode == 'test':
        state_dict = torch.load(args.load_from)
        model.load_state_dict(state_dict)

    if args.mode == 'train':
        optimizer = SGD(model.parameters(),
                        lr=args.lr,
                        momentum=0.9,
                        weight_decay=1e-4,
                        nesterov=True)
        num_iters = args.epochs * (len(train_data_list) // args.batch_size)
        scheduler = CosineAnnealingLR(optimizer, num_iters)

        loss_fn = nn.CrossEntropyLoss().cuda()

        iters_per_epoch = len(train_data_list) // args.batch_size
        for epoch in range(args.epochs):
            model.train()
            loss_rec = SmoothedValue()
            acc_rec = SmoothedValue()
            for batch_idx, (x, label) in enumerate(train_dataloader):
                x, label = x.cuda(), label.cuda()
                output = model(x)
                loss = loss_fn(output, label)
                acc = cal_acc(output.detach().cpu(), label.detach().cpu())
                loss_rec.update(loss)
                acc_rec.update(acc)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                if (batch_idx + 1) % 10 == 0:
                    print(
                        'Epoch: [{:-2d}]  [{:-4d}/{:-4d}] loss: {:.4f}'.format(
                            epoch + 1, batch_idx + 1, iters_per_epoch,
                            loss_rec.avg))
            print('Epoch: [{:-2d}]  loss: {:.4f}  train_acc: {:.2f}%'.format(
                epoch + 1, loss_rec.global_avg, acc_rec.global_avg * 100))

            if args.validation:
                model.eval()
                val_acc_rec = SmoothedValue()
                for x, label in val_dataloader:
                    x = x.cuda()
                    with torch.no_grad():
                        output = model(x)
                    val_acc = cal_acc(output.detach().cpu(), label.detach())
                    val_acc_rec.update(val_acc)
                print('Epoch: [{:-2d}]  val_acc: {:.2f}%'.format(
                    epoch + 1, val_acc_rec.global_avg * 100))

        torch.save(model.state_dict(), os.path.join(args.work_dir, 'temp.pth'))
    elif args.mode == 'test':
        model.eval()
        preds = []
        for x in test_dataloader:
            x = x.cuda()
            with torch.no_grad():
                output = model(x)
            pred = np.argmax(output, axis=1)
            preds.append(pred)
        result_fname = os.path.join(args.work_dir, 'results.txt')
        with open(result_fname, 'w+') as f:
            f.writelines([str(pred) + '\n' for pred in preds])
        print(f'inference results have been saved in {result_fname}')


if __name__ == '__main__':
    main()
