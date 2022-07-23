import argparse
import os

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import FeatureDataset
from model import Classifier
from utils import SmoothedValue, FocalLoss


def get_args():
    parser = argparse.ArgumentParser('AML Assign2 | MF21330035 B.K. Huang',
                                     add_help=False)

    parser.add_argument('--anno', type=str, help='annotation file')
    parser.add_argument('--num_classes',
                        default=6,
                        type=int,
                        help='number of classes')
    parser.add_argument('--in_channels', default=3072, type=int)

    parser.add_argument('--mode',
                        choices=['train', 'test'],
                        default='train',
                        help='running mode')
    parser.add_argument('--validation', action='store_true', default=False)
    parser.add_argument('--val_fold', default=5, type=int)

    parser.add_argument('--work_dir', default='work_dir/', type=str)
    parser.add_argument('--load_from',
                        default='work_dir/train_ckpt.pth',
                        type=str,
                        help='load model parameters')

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--lr',
                        default=1e-3,
                        type=float,
                        help='lr coefficient, init_lr = lr * bs / 64')
    parser.add_argument('--min_lr',
                        default=1e-5,
                        type=float,
                        help='min lr coefficient')
    parser.add_argument('--loss_fn',
                        choices=['ce', 'focal'],
                        default='ce',
                        help='loss function: cross-entropy | focal loss')

    parser.add_argument('--device',
                        choices=['cuda', 'cpu'],
                        default='cpu',
                        help='running mode')

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    # dataset
    anno_file = args.anno
    num_data = len(pd.read_csv(anno_file, delimiter=','))

    if args.mode == 'train':
        if args.validation:
            val_data_list = list(range(num_data // args.val_fold))
            train_data_list = list(range(num_data // args.val_fold, num_data))
            val_dataset = FeatureDataset(anno_file, val_data_list, mode='val')
        else:
            train_data_list = list(range(num_data))

        train_dataset = FeatureDataset(anno_file,
                                       train_data_list,
                                       mode='train')

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True)
        if args.validation:
            val_dataloader = DataLoader(val_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=False)
    elif args.mode == 'test':
        test_data_list = list(range(num_data))
        test_dataset = FeatureDataset(anno_file, test_data_list, mode='test')
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = Classifier(args.num_classes, args.in_channels)
    if args.device == 'cuda':
        model = model.cuda()
    if args.mode == 'test':
        state_dict = torch.load(args.load_from)
        model.load_state_dict(state_dict)

    if args.mode == 'train':
        init_lr = args.lr * args.batch_size / 64
        minimal_lr = args.min_lr * args.batch_size / 64
        optimizer = AdamW(model.parameters(), lr=init_lr, weight_decay=0.05)
        num_iters = args.epochs * (len(train_data_list) // args.batch_size)
        scheduler = CosineAnnealingLR(optimizer, num_iters, eta_min=minimal_lr)

        if args.loss_fn == 'ce':
            loss_fn = nn.CrossEntropyLoss()
        elif args.loss_fn == 'focal':
            loss_fn = FocalLoss()

        if args.device == 'cuda':
            loss_fn = loss_fn.cuda()

        for epoch in range(args.epochs):
            model.train()
            loss_rec = SmoothedValue()
            acc_rec = SmoothedValue()
            for batch_idx, (x, label) in enumerate(train_dataloader):
                if args.device == 'cuda':
                    x, label = x.cuda(), label.cuda()
                output = model(x)

                loss = loss_fn(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # update loss & acc
                pred = np.argmax(output.detach().cpu(), axis=1)
                acc = accuracy_score(label.detach().cpu(), pred)
                loss_rec.update(loss)
                acc_rec.update(acc)

            train_log = 'Epoch: [{:-2d}]  train_loss: {:.4f}  train_acc: {:.2f}%'.format(
                epoch + 1, loss_rec.global_avg, acc_rec.global_avg * 100)

            if args.validation:
                model.eval()
                pred_list = []
                label_list = []
                for x, label in val_dataloader:
                    if args.device == 'cuda':
                        x = x.cuda()
                    with torch.no_grad():
                        output = model(x)

                    pred_list.append(np.argmax(output.detach().cpu(), axis=1))
                    label_list.append(label.detach())

                preds = torch.cat(pred_list, axis=0)
                labels = torch.cat(label_list, axis=0)
                val_acc = accuracy_score(labels, preds)
                macro_f1 = f1_score(labels, preds, average='macro')

                train_log += '  val_acc: {:.2f}%  f1_score: {:.4f}'.format(
                    val_acc * 100, macro_f1)

            print(train_log)

        ckpt_fname = os.path.join(args.work_dir, 'train_ckpt.pth')
        torch.save(model.state_dict(), ckpt_fname)
        print(f'Train done. Checkpoint has been saved in {ckpt_fname}')

    elif args.mode == 'test':
        model.eval()
        preds = []
        for x in test_dataloader:
            if args.device == 'cuda':
                x = x.cuda()
            with torch.no_grad():
                output = model(x)
            pred = np.argmax(output, axis=1)
            preds.append(pred)

        result_fname = os.path.join(args.work_dir, 'test_preds.txt')
        with open(result_fname, 'w+') as f:
            f.writelines([str(int(pred)) + '\n' for pred in preds])
        print(f'Test done. Results have been saved in {result_fname}')


if __name__ == '__main__':
    main()
