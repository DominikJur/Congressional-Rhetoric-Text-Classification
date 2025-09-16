import torch.nn as nn


# Define the RNN-based text classification model
class RNNClassifier(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, num_classes=3, dropout=0.3
    ):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # embedding layer
        self.lstm1 = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=True
        )  # first LSTM layer
        self.lstm2 = nn.LSTM(
            hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True
        )  # second LSTM layer
        self.dropout = nn.Dropout(dropout)  # dropout layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # fully connected layer
        self.softmax = nn.Softmax(dim=1)  # softmax layer

    def forward(self, x):  # forward pass
        x = self.embedding(x)  # embed the input
        out, _ = self.lstm1(x)  # first LSTM layer
        out, _ = self.lstm2(out)  # second LSTM layer
        out = self.dropout(out)  # apply dropout
        out = out[:, -1, :]  # Use the last time step
        logits = self.fc(out)  # fully connected layer
        return self.softmax(logits)  # apply softmax to get probabilities
