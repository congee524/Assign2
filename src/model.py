import torch.nn as nn


class Classifier(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 hidden_dim=128,
                 dropout_ratio=0.5,
                 init_std=0.01):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        self.linear = nn.Linear(in_channels, hidden_dim)
        self.act = nn.ReLU()
        if dropout_ratio > 0:
            self.dropout = nn.Dropout(p=dropout_ratio)
        else:
            self.dropout == nn.Identity()
        self.fc_cls = nn.Linear(hidden_dim, num_classes)

        self.loss_cls = nn.CrossEntropyLoss()

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.linear.weight,
                                mode='fan_out',
                                nonlinearity='relu')
        nn.init.normal_(self.fc_cls.weight, std=self.init_std)

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        x = self.dropout(x)
        cls_score = self.fc_cls(x)
        return cls_score
