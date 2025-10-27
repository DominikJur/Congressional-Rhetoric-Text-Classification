import json
import os

import torch
import pandas as pd

from src.evaluation import evaluate_classification
from src.models import RNNClassifier
from src.training import get_dataloaders, train_rnn_text_classifier_with_deep_oversampling, train_rnn_text_classifier_standard

if __name__ == "__main__":
    # Parameters

    json_path = os.path.join(
        "data", "labeled_text_data.json"
    )  # Path to the labeled dataset
    batch_size = 64
    epochs = 200
    learning_rate = 0.001
    embedding_dim = 50
    hidden_dim = 64
    rnn_layers = 3
    num_classes = 3  # positive negative and neutral
    train = True  # Set to False to skip training and only evaluate
    use_oversampling = False  # Whether to use deep oversampling
    # Device configuration: use GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Load data
    dataloader_train, dataloader_test, minority_classes, vocab_size, weights_matrix, pad_idx = get_dataloaders(
        json_path, batch_size=batch_size, embedding_dim=embedding_dim
    )
    model_name = f"rnn_{epochs}_epoch_{rnn_layers}_layers_{'deep_oversampling' if use_oversampling else 'standard'}.pth"
    
    if train:
        # Initialize model
        model = RNNClassifier(
            hidden_dim=hidden_dim, 
            weights_matrix=weights_matrix,
            pad_idx=pad_idx,
            rnn_layers=rnn_layers,
            num_classes=num_classes
        )
        # Train model (the trainer will move the model to device)
        trained_model = train_rnn_text_classifier_with_deep_oversampling(
            model, 
            dataloader_train, 
            epochs=epochs, 
            learning_rate=learning_rate, 
            minority_classes=minority_classes, 
            dos_k=5, 
            dos_lambda=0
        ) if use_oversampling else train_rnn_text_classifier_standard(
            model, 
            dataloader_train, 
            epochs=epochs, 
            learning_rate=learning_rate
        )
        # Move model to CPU before saving state dict to avoid CUDA tensors in checkpoint
        trained_model_cpu = trained_model.to(torch.device("cpu"))
        torch.save(
            trained_model_cpu.state_dict(),
            os.path.join("models", "rnn_text_classifier_lambda_0.pth"),
        )
    else:
        # Load the trained model
        trained_model = RNNClassifier(
            hidden_dim=hidden_dim, 
            weights_matrix=weights_matrix,
            pad_idx=pad_idx,
            rnn_layers=rnn_layers,
            num_classes=num_classes
        )
        # Load with map_location to ensure correct device
        performance = pd.read_csv('performance.csv')
        performance.sort_values('f1', ascending=False, inplace=True)
        best_model = performance.iloc[0]['name']

        state = torch.load(
            os.path.join("models", best_model), map_location=device
        )
        trained_model.load_state_dict(state)
        trained_model = trained_model.to(device)
    trained_model.eval()  # Set to evaluation mode
    # Evaluate model
    metrics = evaluate_classification(dataloader_test, trained_model)

    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}" if metric != "confusion_matrix" else f"{metric}:\n{value}")

    if train:
        # append performance to csv
        if not os.path.exists('performance.csv'):
            performance = pd.DataFrame(columns=['name', 'f1'])
            performance.to_csv('performance.csv', index=False)
        performance = pd.read_csv('performance.csv')
        f1_score = metrics['f1']
        performance.loc[len(performance)] = {'name': model_name, 'f1': f1_score}
        performance.to_csv('performance.csv', index=False)
