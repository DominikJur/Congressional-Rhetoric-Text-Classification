import torch
import torch.nn as nn


# Define the RNN-based text classification model
class RNNClassifier(nn.Module):
    def __init__(
        self, 
        embedding_dim,  # This will be 300 (from GloVe)
        hidden_dim, 
        weights_matrix,   # <-- NEW
        pad_idx,          # <-- NEW
        num_classes=3, 
        dropout=0.3
    ):
        super(RNNClassifier, self).__init__()
        
        # --- MODIFIED Embedding Layer ---
        # Load the pre-trained weights into the embedding layer
        self.embedding = nn.Embedding.from_pretrained(
            weights_matrix,         # The weights we just loaded
            freeze=False,           # Set to True to *not* fine-tune embeddings
            padding_idx=pad_idx     # Tell the layer which token is for padding
        )
        
        # The rest of your model is the same
        self.lstm1 = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=False
        )
        self.lstm2 = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = self.dropout(out)
        deep_features = out[:, -1, :]  # Use the last time step
        logits = self.fc(deep_features)
        return logits, deep_features