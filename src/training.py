import pandas as pd  # for data manipulation
import torch  # Deep learning framework
import torch.nn as nn  # for neural network modules
import torch.optim as optim  # for optimization algorithms
import torch.nn.functional as F  # for other utilities
import tqdm  # for progress bar
from sklearn.model_selection import train_test_split  # for splitting dataset
from torch.utils.data import DataLoader, TensorDataset  # for creating data loaders
import torchtext.vocab as vocab
from torchtext.data.utils import get_tokenizer
import numpy as np  # for numerical operations
from collections import Counter  # for counting word frequencies

from src.models import RNNClassifier  # Import the RNNClassifier class from models.py


def preprocess_text(text):
    # TODO Insert preprocessing here
    return text


def get_dataloaders(
    json_path, batch_size=64, test_split=0.2, embedding_dim=300
):
    df = pd.read_json(json_path, orient="index")  # read the labeled dataset
    texts = df["transcription"].tolist()
    texts = [preprocess_text(text) for text in texts]
    labels_list = df["label"].tolist()

    # Load tokenizer
    tokenizer = get_tokenizer('basic_english')
    tokenized_texts = [tokenizer(text) for text in texts]
    
    # Create a vocab object from our texts, adding <unk> (unknown) and <pad> tokens
    glove = vocab.GloVe(name='6B', dim=embedding_dim)
    
    # --- START: CODE FOR OLD torchtext 0.6.0 ---
    
    # 1. Build a counter from the tokens
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)
    
    # 2. Build vocab from the counter
    # This is the old way to add special tokens
    vocab_obj = vocab.Vocab(counter, specials=["<unk>", "<pad>"], min_freq=1)
    
    # 3. Set the unk_index manually (replaces set_default_index)
    vocab_obj.unk_index = vocab_obj["<unk>"]
    
    vocab_size = len(vocab_obj)
    pad_idx = vocab_obj["<pad>"]
    
    # 4. Create weights matrix (uses .itos not .get_itos())
    weights_matrix = torch.zeros((vocab_size, embedding_dim))
    for i, token in enumerate(vocab_obj.itos): # <-- Use .itos attribute
        weights_matrix[i] = glove[token]
    
    # 5. Convert texts to indices (uses .stoi not vocab())
    text_indices = [
        [vocab_obj.stoi.get(token, vocab_obj.unk_index) for token in t] 
        for t in tokenized_texts
    ]
    # Convert ID lists to Tensors
    text_tensors = [torch.tensor(t, dtype=torch.long) for t in text_indices]
    
    input_tokens = nn.utils.rnn.pad_sequence(
        text_tensors, batch_first=True, padding_value=pad_idx
    )
    
    labels = torch.tensor(labels_list, dtype=torch.long)

    # Split into train and test sets
    input_tokens_train, input_tokens_test, labels_train, labels_test = train_test_split(
        input_tokens, labels, test_size=test_split, random_state=42
    )

    print(np.unique(labels_list, return_counts=True))


    threshold = max(np.unique(labels_list, return_counts=True)[1])  / len(labels_list) # arbitrary threshold for minority class definition
    
    minority_classes = []
    for cls in np.unique(labels_list):
        cls_ratio = (labels_train == cls).sum().item() / len(labels_train)
        if cls_ratio < threshold:
            minority_classes.append(cls.item())
    print(f"Minority classes: {minority_classes}")

    # Create datasets
    dataset_train = TensorDataset(input_tokens_train, labels_train)
    dataset_test = TensorDataset(input_tokens_test, labels_test)
    # Create dataloaders
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    return dataloader_train, dataloader_test, minority_classes, vocab_size, weights_matrix, pad_idx

def train_rnn_text_classifier_with_deep_oversampling(
    model,
    dataloader_train,
    minority_classes=[],
    dos_k=5,
    dos_lambda=200,
    epochs=100,
    learning_rate=0.001,
    oversampling_technique='deep_feature_SMOTE',
):
    """
        This function is used to train the RNN model using the oversampling technique from the paper 'Deep Over-sampling Framework for Classifying
    Imbalanced Data'. The idea is to use the deep features extracted from the model to generate synthetic samples for the minority classes during training.
    """
    
    match oversampling_technique:
        case 'deep_feature_SMOTE':
            oversampling = deep_feature_SMOTE
        case _:
            raise ValueError(f"Unknown oversampling technique: {oversampling_technique}") 
    
    # make sure the model is an instance of RNNClassifier
    assert isinstance(model, RNNClassifier)

    # device handling: use GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()  # suitable for multi-class classification
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate
    )  # Adam optimizer, state of the art

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_oversample_loss = 0.0
        total_model_loss = 0.0

        for inputs, targets in tqdm.tqdm(
            dataloader_train, desc=f"Training Epoch {epoch+1}/{epochs}"
        ):
            # Move batch tensors to the same device as the model
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()  # zero the parameter gradients
            outputs, deep_features = model(inputs)  # forward pass
            model_loss = criterion(outputs, targets)  # compute loss
            """
            Deep Over-sampling Loss Calculation,
            this might be scary, if so pretend its not there and read after the code block
            """

            loss, avg_oversample_loss = oversampling(
                model_loss, deep_features, targets, minority_classes, k=dos_k, lambda_coeff=dos_lambda
            )

            loss.backward()  # backward pass
            optimizer.step()  # update weights
            
            total_model_loss += model_loss.item()
            total_oversample_loss += avg_oversample_loss.item()
        # Print combined loss for the epoch
        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Class Loss: {total_model_loss / len(dataloader_train):.4f}, "
            f"DOS Loss: {total_oversample_loss / len(dataloader_train):.4f}"
        )

    return model




def deep_feature_SMOTE(model_loss, deep_features, targets, minority_classes, k=5, lambda_coeff=0.5):
    """
    Deep Over-sampling Loss Calculation.
    Args:
        model_loss: The original loss from the model (CrossEntropyLoss).
        deep_features: The deep features extracted from the model.
        targets: The true labels for the batch.
        minority_classes: List of classes considered as minority.
        k: Number of nearest neighbors to consider.
        lambda_coeff: Weighting factor for the oversampling loss.
    Returns:
        Combined loss: model_loss + lambda_coeff * oversampling_loss
    """
    criterion_oversample = nn.MSELoss()  # for oversampling loss
    device = deep_features.device
    oversample_loss = torch.tensor(0.0, device=device)
    if minority_classes:
        for cls in minority_classes:
            cls_ids = (
                (targets == cls).nonzero(as_tuple=False).squeeze(-1)
            )  # Find indices of the *current* minority class
            if len(cls_ids) < k + 1:
                continue  # Skip if too little of this class in the batch
            minority_deep_features = deep_features[
                cls_ids
            ]  # Extract deep features for the minority class

            # now find the k nearest neighbors for each sample in minority_deep_features
            # 1. Expand tensors for broadcasting
            # X1 shape: [num_minority, 1, features]
            X1 = minority_deep_features.unsqueeze(1)
            # X2 shape: [1, num_minority, features]
            X2 = minority_deep_features.unsqueeze(0)
            
            # 2. Calculate squared differences
            # diff shape: [num_minority, num_minority, features]
            diff = X1 - X2
            
            # 3. Sum along the feature dimension to get squared L2 dist
            # dists_sq shape: [num_minority, num_minority]
            dists_sq = torch.sum(diff**2, dim=2)
            _, knn_indices = torch.topk(
                    dists_sq, k=k + 1, largest=False
            )  # Get indices of k nearest neighbors (including self)
            knn_indices = knn_indices[:, 1:]  # Exclude self

            # Generate synthetic targets
            rand_neighbors = torch.randint(
                0, k, (len(cls_ids),), device=device
            )
            selected_neighbors = knn_indices[
                torch.arange(len(cls_ids)), rand_neighbors
            ]
            neighbour_deep_features = minority_deep_features[selected_neighbors]
            gamma = torch.rand(len(cls_ids), 1, device=device)
            synthetic_deep_features = (
                gamma * minority_deep_features
                + (1 - gamma) * neighbour_deep_features
            )

            # Compute oversampling loss
            oversample_loss += criterion_oversample(
                minority_deep_features, synthetic_deep_features.detach()
            )

    avg_oversample_loss = (
        oversample_loss / len(minority_classes)
        if minority_classes
        else torch.tensor(0.0, device=device)
    )
    loss = model_loss + lambda_coeff * avg_oversample_loss  # combined loss
    return loss, avg_oversample_loss
    