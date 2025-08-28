import torch.nn as nn


class MlpNet(nn.Module):
    def __init__(self, hidden : list):
        super(MlpNet, self).__init__()
        self.hidden = hidden
        self.n_layer = len(self.hidden) - 1
        self.layers = nn.ModuleList()
        for layer in range(self.n_layer):
            # if last layer, do not use non-linear activation function
            if layer == self.n_layer - 1:
                self.layers.append(nn.Sequential(
                    nn.Linear(self.hidden[layer], self.hidden[layer + 1]),
                ))
            else:
                self.layers.append(nn.Sequential(
                    nn.Linear(self.hidden[layer], self.hidden[layer + 1]),
                    nn.ReLU(),
                ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class fusionLayer(nn.Module):
    def __init__(self, hidden):
        super(fusionLayer, self).__init__()
        self.model = MlpNet(hidden).float()

    def forward(self, emb):
        fused = self.model.forward(emb)
        return fused
