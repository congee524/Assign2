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

        self.stem_layer = nn.Linear(in_channels, hidden_dim, bias=False)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU()

        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4, bias=False)
        self.norm1 = nn.BatchNorm1d(hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim, bias=False)
        self.norm2 = nn.BatchNorm1d(hidden_dim)

        if dropout_ratio > 0:
            self.dropout = nn.Dropout(p=dropout_ratio)
        else:
            self.dropout == nn.Identity()
        self.fc_cls = nn.Linear(hidden_dim, num_classes)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.stem_layer.weight,
                                mode='fan_out',
                                nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc1.weight,
                                mode='fan_out',
                                nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight,
                                mode='fan_out',
                                nonlinearity='relu')
        nn.init.normal_(self.fc_cls.weight, std=self.init_std)

    def forward(self, x):
        x = self.stem_layer(x)
        x = self.act(self.norm(x))

        identity = x
        x = self.norm1(self.fc1(x))
        x = self.act(x)
        x = self.norm2(self.fc2(x))
        x = x + identity
        x = self.act(x)

        x = self.dropout(x)
        cls_score = self.fc_cls(x)
        return cls_score
