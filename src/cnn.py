import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_classes=3,
                 num_filters=100, kernel_sizes=(3, 4, 5), dropout=0.4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        e = self.embed(x).transpose(1, 2)                    # (B, E, L)
        feats = [torch.relu(c(e)).max(dim=2).values for c in self.convs]
        h = self.dropout(torch.cat(feats, dim=1))
        return self.fc(h)