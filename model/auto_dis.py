import torch
import torch.nn as nn

class AutoDis(nn.Module):
    def __init__(self, meta_embedding_vocab_size:int, dims:int, pad_index:torch.float32=-1):
        super(AutoDis, self).__init__()
        self.temperature = torch.nn.Parameter(torch.tensor(10), requires_grad=False)
        self.skip_factor = torch.nn.Parameter(torch.tensor(0.3), requires_grad=False)
        self.pad_index = torch.nn.Parameter(torch.tensor(pad_index), requires_grad=False)
        self.meta_embedding = torch.nn.Parameter(torch.randn(meta_embedding_vocab_size, dims))
        self.linear = nn.Sequential(
            nn.Linear(1, meta_embedding_vocab_size),
            nn.LeakyReLU(),
        )
        self.W = nn.Sequential(nn.Linear(meta_embedding_vocab_size, meta_embedding_vocab_size))

    def forward(self, x:torch.tensor):
        self.eval()
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)
        if len(x.shape) == 0:
            x = x.view(1, -1)
        h = self.linear(x.float())
        #print(h)
        xj = self.W(h) + self.skip_factor * h
        #print(xj)
        logits = xj / self.temperature
        #print(logits)
        weights = logits.softmax(dim=-1)
        #print(weights)
        out = torch.mm(weights, self.meta_embedding)
        #print(x, self.pad_index, x==self.pad_index)
        mask = (x==self.pad_index).expand((-1, self.meta_embedding.shape[-1]))
        out.masked_fill_(mask, 0)
        #print(self.meta_embedding,out)
        #print(self.meta_embedding[0] * weights[0][0] + self.meta_embedding[1] * weights[0][1] + self.meta_embedding[2] * weights[0][2] - out)
        return out
