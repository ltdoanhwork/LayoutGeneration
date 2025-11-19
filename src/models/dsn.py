from __future__ import annotations
import torch
import torch.nn as nn

class EncoderFC(nn.Module):
    """Map precomputed frame features (T,D) to hidden (T,H)."""
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()

    def forward(self, x):  # x: (B,T,D)
        return self.act(self.fc(x))

class DSNPolicy(nn.Module):
    """Bi-LSTM -> per-frame Bernoulli probability."""
    def __init__(self, hidden_dim: int, lstm_hidden: int, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=lstm_hidden,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(lstm_hidden*2, 1)

    def forward(self, h):  # h: (B,T,H)
        out,_ = self.lstm(h)        # (B,T,2H_lstm)
        out = self.drop(out)
        logits = self.head(out)     # (B,T,1)
        probs = torch.sigmoid(logits).squeeze(-1)  # (B,T)
        return probs
