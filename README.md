# Congressional-Rhetoric-Text-Classification

This README documents the repository structure, how to set up the environment, how to transcribe videos, how to train and evaluate the model and data formats.

## Table of contents

- Overview
- Project structure
- Requirements
- Setup
- Running transcription (Whisper)
- Creating labeled dataset
- Training the RNN classifier
- Evaluating the model
- Data formats
- Troubleshooting
- Developer notes

## Overview

The pipeline consists of two main parts:

- Transcription: `transcription_runner.py` — extracts audio from MP4 files, runs Whisper to produce text and timestamped chunks, and writes a labeled JSON dataset.
- Modeling: `src/` — contains code to create dataloaders, a simple RNN classifier, training loop, and evaluation utilities.

The code is written to use a GPU automatically when available..

## Project structure

- `transcription_runner.py` — create transcriptions from `data/*.mp4` and produce `data/labeled_text_data.json` using labels in `data/labels.csv`.
- `main_rnn.py` — entry point to train or evaluate the RNN classifier. Reads `data/labeled_text_data.json` and uses files in `models/` to load/save checkpoints.
- `data/` — store MP4 files, labels and generated datasets here. 
- `models/` — trained model files are saved here.
- `src/` — Python package with the project logic:
	- `__init__.py`
	- `constants.py` — constant mappings such as `classes_dict`.
	- `training.py` — dataloader creation and training loop (device-aware).
	- `models.py` — model definitions (RNN and some experimental transformer blocks).
	- `evaluation.py` — evaluation utilities and metrics wrapper.

## Requirements

- Python 3.11+
- `PyTorch` (compatible with your CUDA if using GPU)
- `Hugging Face Transformers`
- `pandas`, `scikit-learn`, `tqdm`, `pydub`, `numpy`

Example install:

```bash
python -m venv venv 
venv\Scripts\activate
pip install --upgrade pip
pip install torch torchtext --index-url https://download.pytorch.org/whl/cu118
pip install transformers pandas scikit-learn tqdm pydub numpy
```

## Running transcription

The transcription pipeline extracts audio from MP4 files and runs Whisper.
Basic usage (from repo root):

```bash
python transcription_runner.py
```

This will:

- Look for `*.mp4` files in the `data/` directory.
- For each file, extract audio to a temporary WAV and call the Whisper pipeline.
- Create `data/labeled_text_data.json` when `create_labeled_dataset()` is run (it requires `data/labels.csv` to exist).

## Creating the labeled dataset

`transcription_runner.py` includes `create_labeled_dataset(transcriptions, labels_file, output_file)` which:

- Reads `data/labels.csv` that must have columns: `filename`, `label`.
- Matches filenames (basename) to transcriptions.
- Writes `data/labeled_text_data.json` (index keyed by filename) with fields `transcription` (text), `timestamped_chunks`, and `label`.

Call the transcription and dataset creation as shown in the `if __name__ == '__main__'` block of `transcription_runner.py`.

## Data formats

- `data/labels.csv` — expected columns: `filename`, `label` where `filename` is the original MP4 filename or a path.
- `data/labeled_text_data.json` — written in `orient='index'` JSON format. Each key is the filename and its value is an object with these fields:

	```json
	"578982906.mp4": {
		"transcription": "Full transcribed text...",
		"timestamped_chunks": [ ... ],
		"label": 0
	}
	```

	Chunk format

	The transcription pipeline returns timestamped chunks (segments) alongside the full transcription text. The code normalizes different pipeline return formats into a `chunks` (or `timestamped_chunks` in the output JSON) array. Each chunk is a small JSON object with at least the following fields:

	- `timestamp`: list of types float — start time in seconds from the beginning of the audio and the end time in seconds
	- `text`: string — the transcribed text for this time span

	Example of a `timestamped_chunks` value in the JSON output:

	```json
	"timestamped_chunks": [
		{"timestamp": [0.0, 2.1], "text": "Good morning everyone."},
		{"timestamp": [2.1, 5.5], "text": "I'm here to discuss the new policy."},
		{"timestamp": [5.5, 7.0], "text": "Thank you."}
	]
	```

	
- `src/training.get_dataloaders(json_path)` expects the labeled JSON in `orient='index'` format and will read it using `pd.read_json(json_path, orient='index')`. It tokenizes `transcription` using a Hugging Face tokenizer and returns train/test DataLoaders.

## Training the RNN classifier

`main_rnn.py` is the entry point for training and evaluating the RNN classifier.

Key options in `main_rnn.py`:

- `json_path` — path to labeled JSON file (default `data/labeled_text_data.json`).
- `train` — when `True`, the script trains and saves the model; when `False`, it loads a saved model and evaluates.

Example: train and save:

```powershell
python main_rnn.py
```

Important implementation notes:

- The training code (`src/training.py`) automatically selects device = GPU (cuda) if available and moves the model and batch tensors to that device.
- When a model is saved after training, it is moved to CPU first before `torch.save(model.state_dict(), ...)` to produce portable checkpoints.

## Model and evaluation

- `src/models.py` contains `RNNClassifier`, a two-layer LSTM-based classifier that expects token ids (from a tokenizer) as input, and returns class probabilities.
- `src/evaluation.py` contains `ClassificationBenchmark` and `evaluate_classification()` which compute accuracy, precision, recall, F1, Matthews Correlation Coefficient, informedness, markedness, and confusion matrix. The evaluator moves batches to the model device before forward passes and converts tensors to CPU before numpy operations.

## Deep Feature SMOTE

Our model addresses class imbalance using a specialized helper function called `deep_feature_SMOTE`, which implements the "Deep Over-sampling" (DOS) framework. Instead of oversampling the raw text, this function operates directly on the **deep feature space**—the rich, 512-dimensional hidden vectors produced by the RNN's LSTM layers. For each training batch, the function identifies all samples belonging to the designated minority classes and finds their k-nearest neighbors within this feature space. It then generates a synthetic "target" feature by interpolating between a minority sample and one of its neighbors. A secondary **MSE loss** is calculated to pull the original sample's feature vector closer to this new synthetic target. This encourages the model to learn a more compact and discriminative representation for minority classes, making them easier to separate from the majority classes in the final classification layer. This DOS loss is weighted by a `lambda` coefficient and added to the primary `CrossEntropyLoss` for a combined backpropagation step.