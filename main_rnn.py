import os

import torch

from src.evaluation import evaluate_classification
from src.models import RNNClassifier
from src.training import get_dataloaders, train_rnn_text_classifier

if __name__ == "__main__":
    # Parameters
    csv_path = os.path.join(
        "data", "labeled_text_data.csv"
    )  # Path to the labeled dataset
    batch_size = 64
    epochs = 30
    learning_rate = 0.001
    vocab_size = 30522  # Typical size for BERT tokenizer
    embedding_dim = 128
    hidden_dim = 256
    num_classes = 3  # Adjust based on your dataset
    train = False  # Set to False to skip training and only evaluate
    # Load data
    dataloader_train, dataloader_test = get_dataloaders(csv_path, batch_size=batch_size)
    if train:
        # Initialize model
        model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, num_classes)

        # Train model
        trained_model = train_rnn_text_classifier(
            model, dataloader_train, epochs=epochs, learning_rate=learning_rate
        )
        # Save the trained model
        torch.save(
            trained_model.state_dict(),
            os.path.join("models", "rnn_text_classifier.pth"),
        )
    else:
        # Load the trained model
        trained_model = RNNClassifier(
            vocab_size, embedding_dim, hidden_dim, num_classes
        )
        trained_model.load_state_dict(
            torch.load(os.path.join("models", "rnn_text_classifier.pth"))
        )
    trained_model.eval()  # Set to evaluation mode
    # Evaluate model
    metrics = evaluate_classification(dataloader_test, trained_model)

    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
