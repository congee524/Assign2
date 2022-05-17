import torch
import pandas as pd
from torch.utils.data import Dataset


class FeatureDataset(Dataset):

    def __init__(self, anno_file, data_list, mode='train'):
        assert mode in ['train', 'val', 'test']

        self.anno_file = anno_file
        self.data_list = data_list
        self.mode = mode

        self.feat_data = torch.load(self.anno_file[:-4] + '_feat_embed.pt',
                                    map_location='cpu')

        if self.mode != 'test':
            self.label_array = list(
                pd.read_csv(anno_file, delimiter=',')['target'])

    def __getitem__(self, index):
        if self.mode == 'train' or self.mode == 'val':
            idx = self.data_list[index]
            data = self.feat_data[idx]
            label = self.label_array[idx]
            return data, int(label)
        elif self.mode == 'test':
            idx = self.data_list[index]
            data = self.feat_data[idx]
            return data

    def __len__(self):
        return len(self.data_list)
