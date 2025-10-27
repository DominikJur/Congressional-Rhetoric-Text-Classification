import re
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
from plotly.subplots import make_subplots
import numpy as np  # for numerical operations
from collections import Counter  # for counting word frequencies

from src.models import RNNClassifier  # Import the RNNClassifier class from models.py
import plotly.express as px  # for visualizations
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    """
    A proper text preprocessing function:
    1. Lowercase
    2. Remove punctuation and numbers
    3. Remove stopwords
    4. Apply lemmatization
    """
    import nltk
    nltk.download('stopwords');
    nltk.download('wordnet');
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    
    tokens = text.split()
    
    # --- CHANGED ---
    # Apply lemmatization instead of stemming
    processed_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
    ]
    
    return " ".join(processed_tokens)

def get_dataloaders(
    json_path, batch_size=64, test_split=0.2, embedding_dim=300
):
    df = pd.read_json(json_path, orient="index")  # read the labeled dataset
    texts = df["transcription"].tolist()
    texts = [preprocess_text(text) for text in texts]
    labels_list = df["label"].tolist()

    tokenizer = get_tokenizer('basic_english')
    tokenized_texts = [tokenizer(text) for text in texts]
    
    glove = vocab.GloVe(name='6B', dim=embedding_dim)
    
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)
    
    vocab_obj = vocab.build_vocab_from_iterator(
        tokenized_texts, 
        specials=["<unk>", "<pad>"]
    )
    vocab_obj.set_default_index(vocab_obj["<unk>"])
    vocab_size = len(vocab_obj)
    pad_idx = vocab_obj["<pad>"]
    
    weights_matrix = torch.zeros((vocab_size, embedding_dim))
    for i, token in enumerate(vocab_obj.get_itos()): # get_itos() returns list of tokens in vocab
        weights_matrix[i] = glove[token]
    
    text_indices = [vocab_obj(t) for t in tokenized_texts]
    
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

def train_rnn_text_classifier_standard(
    model,
    dataloader_train,
    epochs=100,
    learning_rate=0.001,
):
    """
    This function is used to train the RNN model using standard training procedure without any oversampling.
    """
    # make sure the model is an instance of RNNClassifier
    assert isinstance(model, RNNClassifier)
    # device handling: use GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(f"Using device: {device}")
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()  # suitable for multi-class classification
    embedding_params = model.embedding.parameters()
    
    # rest of the model's parameters
    rnn_params = [
        p for n, p in model.named_parameters() 
        if "embedding" not in n and p.requires_grad
    ]
    optimizer = optim.AdamW(
        [
            {'params': rnn_params, 'lr': learning_rate}, # e.g., 0.001
            {'params': embedding_params, 'lr': learning_rate / 20} # e.g., 0.00005
        ],
        weight_decay=0.01
    ) # Adam optimizer, state of the art

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    model_loss_history = []

    # Training loop
    model.train()
    for epoch in range(epochs):
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

            model_loss_history.append(model_loss.item())

            model_loss.backward()  # backward pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
            optimizer.step()  # update weights
            
            total_model_loss += model_loss.item()
        # Print combined loss for the epoch
        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Class Loss: {total_model_loss / len(dataloader_train):.4f}"
        )
        scheduler.step()  # update learning rate


    plot_loss_curves(model_loss_history)

    return model



def plot_loss_curves(model_loss_history, oversample_loss_history=[]):
    """
    Plots the loss curves as two separate subplots (no shared y-axis).
    Args:
        model_loss_history: List of model loss values over training iterations.
        oversample_loss_history: List of oversampling loss values over training iterations.
    """
    iterations = list(range(len(model_loss_history)))
    if len(iterations) > 100:
        iterations = iterations[::5]
        model_loss_history = model_loss_history[::5]
        oversample_loss_history = oversample_loss_history[::5]
    if not oversample_loss_history:
        fig = px.line(
            x=iterations, y=model_loss_history,
            labels={'x': 'Iteration', 'y': 'Loss'},
        )
        fig.update_xaxes(title_text="Iteration")
        fig.update_yaxes(title_text="Loss")
        
    else:
        import plotly.graph_objects as go

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_yaxes=False,
            vertical_spacing=0.3,
            subplot_titles=("Model Loss", "Oversampling Loss"),
        )

        fig.add_trace(
            go.Scatter(x=iterations, y=model_loss_history, mode="lines", name="Model Loss"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=iterations, y=oversample_loss_history, mode="lines", name="Oversampling Loss"
            ),
            row=2,
            col=1,
        )

        fig.update_xaxes(title_text="Iteration", row=1, col=1)
        fig.update_xaxes(title_text="Iteration", row=2, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=2, col=1)

    fig.update_layout(title_text="Training Loss Curves", height=600, showlegend=False)
    fig.show()
    return fig

"""
Beware the code below, it might be scary.
It implements the deep oversampling technique from the paper:
'Deep Over-sampling Framework for Classifying Imbalanced Data'.
We ended up not using it so you can safely ignore it.
"""




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
            oversampling =  no_oversampling
    # make sure the model is an instance of RNNClassifier
    assert isinstance(model, RNNClassifier)
    if dos_lambda == 0:
        oversampling = no_oversampling
    # device handling: use GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(f"Using device: {device}")
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()  # suitable for multi-class classification
    embedding_params = model.embedding.parameters()
    
    # rest of the model's parameters
    rnn_params = [
        p for n, p in model.named_parameters() 
        if "embedding" not in n and p.requires_grad
    ]
    optimizer = optim.AdamW(
        [
            {'params': rnn_params, 'lr': learning_rate}, # e.g., 0.001
            {'params': embedding_params, 'lr': learning_rate / 20} # e.g., 0.00005
        ],
        weight_decay=0.01
    ) # Adam optimizer, state of the art
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    model_loss_history = []
    oversample_loss_history = []


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

            model_loss_history.append(model_loss.item())
            oversample_loss_history.append(avg_oversample_loss.item())

            loss.backward()  # backward pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
            optimizer.step()  # update weights
            
            total_model_loss += model_loss.item()
            total_oversample_loss += avg_oversample_loss.item()
        # Print combined loss for the epoch
        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Class Loss: {total_model_loss / len(dataloader_train):.4f}, "
            f"DOS Loss: {total_oversample_loss / len(dataloader_train):.4f}"
        )
        scheduler.step()  # update learning rate


    plot_loss_curves(model_loss_history, oversample_loss_history)

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
    
def no_oversampling(model_loss, deep_features, targets, minority_classes, k=5, lambda_coeff=0.5):
    """
    No oversampling, returns the original model loss.
    """
    device = deep_features.device
    avg_oversample_loss = torch.tensor(0.0, device=device)
    return model_loss, avg_oversample_loss