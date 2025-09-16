import pandas as pd  # for data manipulation
import torch  # Deep learning framework
import torch.nn as nn  # for neural network modules
import torch.optim as optim  # for optimization algorithms
import tqdm  # for progress bar
from sklearn.model_selection import train_test_split  # for splitting dataset
from torch.utils.data import (DataLoader,  # for creating data loaders
                              TensorDataset)
from transformers import AutoTokenizer  # for tokenization

from src.models import \
    RNNClassifier  # Import the RNNClassifier class from models.py


def get_dataloaders(
    csv_path, batch_size=64, tokenizer_name="bert-base-uncased", test_split=0.2
):
    df = pd.read_csv(csv_path)
    texts = df["transcription"].tolist()
    labels_list = df["label"].tolist()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Tokenize and pad sequences
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    input_ids = encoded["input_ids"]
    labels = torch.tensor(labels_list, dtype=torch.long)

    # Split into train and test sets
    input_ids_train, input_ids_test, labels_train, labels_test = train_test_split(
        input_ids, labels, test_size=test_split, random_state=42
    )
    # Create datasets
    dataset_train = TensorDataset(input_ids_train, labels_train)
    dataset_test = TensorDataset(input_ids_test, labels_test)
    # Create dataloaders
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    return dataloader_train, dataloader_test


def train_rnn_text_classifier(model, dataloader_train, epochs=100, learning_rate=0.001):
    # make sure the model is an instance of RNNClassifier
    assert isinstance(model, RNNClassifier)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()  # suitable for multi-class classification
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate
    )  # Adam optimizer, state of the art

    # Training loop
    model.train()
    for epoch in range(epochs):
        for inputs, targets in tqdm.tqdm(
            dataloader_train, desc=f"Training Epoch {epoch+1}/{epochs}"
        ):
            optimizer.zero_grad()  # zero the parameter gradients
            outputs = model(inputs)  # forward pass
            loss = criterion(outputs, targets)  # compute loss
            loss.backward()  # backward pass
            optimizer.step()  # update weights
        print(
            f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}"
        )  # print loss for each epoch

    return model
