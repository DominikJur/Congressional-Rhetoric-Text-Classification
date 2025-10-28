import torch
import torch.nn as nn


# Define the RNN-based text classification model
class RNNClassifier(nn.Module):
    def __init__(
        self, 
        weights_matrix,
        hidden_dim, 
        num_classes,
        pad_idx,
        dropout=0.5,
        rnn_layers=2,
        bidirectional=True
    ):
        super(RNNClassifier, self).__init__()
        
        vocab_size, embedding_dim = weights_matrix.shape
        
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        self.num_directions = 2 if bidirectional else 1
        
        self.embedding = nn.Embedding.from_pretrained(
            weights_matrix, 
            freeze=True,
            padding_idx=pad_idx
        )
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        
        output, (hidden, cell) = self.lstm(embedded)

        if self.num_directions == 2:
            deep_features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            deep_features = hidden[-1]
            
        dropped_features = self.dropout(deep_features)
        
        logits = self.fc(dropped_features)
        
        return logits, deep_features