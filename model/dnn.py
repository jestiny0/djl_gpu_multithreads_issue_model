import torch.nn as nn
from .swi_glu import SwiGLU


class DNN(nn.Module):
    def __init__(self, shape):
        super(DNN, self).__init__()

        dnn_layers = []
        for i in range(len(shape) - 1):
            dnn_layers.extend([
                nn.Linear(shape[i], shape[i+1] * 2),
                SwiGLU(),
                nn.BatchNorm1d(shape[i+1]),
                nn.Dropout(0),
            ])
        dnn_layers.extend([
            nn.Linear(shape[-1], 2),
            nn.LogSoftmax(dim=-1),
        ])
        self.net = nn.Sequential(*dnn_layers)   

    def forward(self, x):
        return self.net(x)
