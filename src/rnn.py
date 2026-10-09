import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden=96,
                 num_classes=3, num_layers=1, dropout=0.4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(self.embed(x))
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(self.dropout(h))