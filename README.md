# Congressional-Rhetoric-Text-Classification

## Installation

### Install uv
```bash
pip install uv
```

### Create virtual environment
```bash
uv venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

### Install dependencies
```bash
uv pip install torch transformers datasets accelerate librosa soundfile scikit-learn numpy pandas tqdm ffmpeg-python
```

### Install system ffmpeg
```bash
sudo apt install ffmpeg  # Ubuntu/Debian
brew install ffmpeg      # macOS
winget install FFmpeg    # Windows
```
