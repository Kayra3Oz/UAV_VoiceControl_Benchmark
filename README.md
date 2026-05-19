# UAV VoiceControl Benchmark

> **[Deutsche Version](README.de.md)**

A benchmark comparing three different voice-controlled UAV pipelines on a **DJI Tello** drone. The project evaluates the trade-off between latency, accuracy, and system complexity across three distinct approaches: a lightweight CNN classifier, and two LLM-based pipelines using Whisper for speech-to-text and LLaMA for command extraction.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Model A — CNN (Keyword Spotting)](#model-a--cnn-keyword-spotting)
  - [Model B — Whisper + LLaMA 3.2:3b](#model-b--whisper--llama-3b)
  - [Model C — Whisper + LLaMA 3.1:8b](#model-c--whisper--llama-8b)
  - [inference.py — Test without Drone](#inferencepy--test-without-drone)
- [Commands Reference](#commands-reference)
- [Logging & Latency Measurement](#logging--latency-measurement)
- [Notebooks](#notebooks)
- [Results](#results)

---

## Project Overview

This project benchmarks three voice control pipelines for UAV navigation:

| | Model A | Model B | Model C |
|---|---|---|---|
| **Approach** | CNN Keyword Spotting | Whisper + LLaMA 3.2:3b | Whisper + LLaMA 3.1:8b |
| **Recording Duration** | 1 second | 5 seconds | 3 seconds |
| **Speech Recognition** | Mel Spectrogram → CNN | OpenAI Whisper (base) | OpenAI Whisper (base) |
| **Command Extraction** | Direct classification | LLaMA via Ollama API | LLaMA via Ollama API |
| **Supports Parameters** | No (fixed distances) | Yes (e.g. "forward 80 cm") | Yes (e.g. "forward 80 cm") |
| **Internet Required** | No | No (runs locally) | No (runs locally) |

---

## Architecture

### Model A — CNN

The CNN architecture is based on the Stanford CS229 paper by Li & Zhou (2017), adapted for this project:

- **Input:** Log-Mel spectrogram of shape `(98, 64, 1)` — derived from 1 second of audio at 16 kHz
- **Architecture:** 3 convolutional layers with BatchNormalization, MaxPooling, and Dropout, followed by a Dense output layer
- **Output:** 12-class softmax — 10 UAV keywords + `silence` + `unknown`
- **Training data:** Google Speech Commands Dataset with augmentation for underrepresented classes

**Preprocessing pipeline:**
1. Record 1 second of audio at 16 kHz
2. Center-pad or trim to exactly 16,000 samples
3. Compute log-Mel spectrogram (n_mels=64, n_fft=400, hop_length=160, fmax=8000)
4. Per-sample normalization (mean=0, std=1)
5. CNN inference → predicted label + confidence

### Model B & C — Whisper + LLaMA

These pipelines use a two-stage approach:

1. **Whisper (base):** Transcribes the recorded audio to text in English
2. **LLaMA via Ollama:** Receives the transcription and extracts a structured drone command (JSON) from natural language

This allows free-form natural language commands such as *"please fly forward fifty centimeters"*, which the LLM maps to `{"command": "forward", "parameter": 50}`.

Model B uses `llama3.2:3b` (faster, lower resource usage), Model C uses `llama3.1:8b` (more capable, higher latency).

---

## Dataset

This project uses the **Google Speech Commands Dataset v2**.

**Download:** [https://www.tensorflow.org/datasets/catalog/speech_commands](https://www.tensorflow.org/datasets/catalog/speech_commands)

Direct download link:
```
http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
```

After downloading, extract the dataset and place it at:
```
data/raw/speech_command_data/
```

The directory should contain one folder per word, e.g.:
```
data/raw/speech_command_data/
├── forward/
├── backward/
├── up/
├── down/
├── stop/
├── right/
├── left/
├── go/
├── one/
├── two/
├── _silence_/
├── _unknown_/
└── ...
```

**Classes used (12 total):**
`forward`, `backward`, `up`, `down`, `stop`, `right`, `left`, `go`, `one`, `two`, `silence`, `unknown`

**Preprocessing steps** (see `notebooks/preprocessing.ipynb`):
- **Augmentation** of underrepresented classes `forward` and `backward` using pitch shift (±2 semitones) and time shift (±10%)
- **Downsampling** of the `unknown` class to balance the dataset
- All clips are center-padded or trimmed to exactly **1 second (16,000 samples)**

---

## Project Structure

```
UAV_VoiceControl_Benchmark/
│
├── drone_control_model_a.py    # CNN-based drone control
├── drone_control_model_b.py    # Whisper + LLaMA 3.2:3b drone control
├── drone_control_model_c.py    # Whisper + LLaMA 3.1:8b drone control
├── inference.py                # Standalone CNN test tool (no drone needed)
├── requirements.txt            # Python dependencies
│
├── notebooks/
│   ├── data_exploration.ipynb  # Dataset analysis and visualization
│   ├── preprocessing.ipynb     # Data augmentation and split preparation
│   └── model_a.ipynb           # CNN training and evaluation
│
├── models/
│   └── model_a.keras           # Trained CNN model (not in repository)
│
├── data/                       # Not in repository — see Dataset section
│   ├── raw/
│   ├── augmented/
│   ├── spectrograms/
│   └── splits/
│
├── results/
│   ├── model_a_confusion_matrix.png
│   └── model_a_training_curves.png
│
└── logs/                       # Auto-generated at runtime
    ├── model_a_YYYYMMDD_HHMMSS.log
    ├── model_b_YYYYMMDD_HHMMSS.log
    ├── model_c_YYYYMMDD_HHMMSS.log
    ├── latencies_model_a.csv
    ├── latencies_model_b.csv
    └── latencies_model_c.csv
```

---

## Requirements

### Python Dependencies

Install via:
```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```
librosa
numpy
matplotlib
scikit-learn
tensorflow
sounddevice
djitellopy
tqdm
pandas
requests
openai-whisper
```

### Ollama (for Model B and C only)

Ollama is not a Python package — it is a standalone application that runs LLMs locally.

1. Download and install Ollama from [https://ollama.com](https://ollama.com)
2. Pull the required models:

```bash
# For Model B
ollama pull llama3.2:3b

# For Model C
ollama pull llama3.1:8b
```

3. Make sure Ollama is running before starting Model B or C:
```bash
ollama serve
```

### Hardware

- **DJI Tello** drone (required for drone control modules)
- Microphone (built-in or external)
- Python 3.9 or higher recommended

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Kayra3Oz/UAV_VoiceControl_Benchmark.git
cd UAV_VoiceControl_Benchmark

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Download dataset (see Dataset section above)
#    Place it at: data/raw/speech_command_data/

# 5. Run preprocessing notebooks (in order)
#    notebooks/preprocessing.ipynb
#    notebooks/model_a.ipynb

# 6. (Optional) Install Ollama for Model B / C
#    https://ollama.com
#    ollama pull llama3.2:3b
#    ollama pull llama3.1:8b
```

---

## Usage

> **Important:** For all drone control modules, connect your computer to the **DJI Tello Wi-Fi network** before running.

---

### Model A — CNN (Keyword Spotting)

The fastest pipeline. Speaks a single keyword — the CNN classifies it directly.

```bash
python drone_control_model_a.py
```

**How it works:**
1. The script loads the trained CNN model (`models/model_a.keras`)
2. Press **Enter** to start a 1-second recording
3. The audio is preprocessed into a Mel spectrogram and fed into the CNN
4. The predicted command is sent to the drone
5. Press **q + Enter** to quit

**Supported commands (speak exactly these words):**

| Word | Drone Action |
|------|-------------|
| `forward` | Fly forward 100 cm |
| `backward` | Fly backward 100 cm |
| `up` | Fly up 30 cm |
| `down` | Fly down 30 cm |
| `left` | Fly left 30 cm |
| `right` | Fly right 30 cm |
| `go` | Take off |
| `stop` | Land |
| `one` | Rotate clockwise 45° |
| `two` | Rotate counter-clockwise 45° |

---

### Model B — Whisper + LLaMA 3.2:3b

Natural language pipeline. Speak full sentences — the LLM extracts the command.

**Prerequisites:** Ollama must be running with `llama3.2:3b` pulled.

```bash
# Terminal 1 — Start Ollama
ollama serve

# Terminal 2 — Start drone control
python drone_control_model_b.py
```

**How it works:**
1. Press **Enter** to start a 5-second recording
2. Whisper transcribes your speech to text
3. LLaMA 3.2:3b extracts a structured command from the transcription
4. The command + parameter is sent to the drone
5. Press **q + Enter** to quit

**Example phrases:**

| You say | Extracted command | Parameter |
|---------|-------------------|-----------|
| "fly forward fifty centimeters" | `forward` | 50 |
| "go up thirty" | `up` | 30 |
| "rotate to the left" | `rotate left` | 30 |
| "stop" | `stop` | 0 |
| "take off" | `start` | 0 |

**Supported commands:** `forward`, `backward`, `right`, `left`, `up`, `down`, `start`, `stop`, `rotate left`, `rotate right`

If no parameter is given, the default is **30 cm**.

---

### Model C — Whisper + LLaMA 3.1:8b

Same pipeline as Model B but with the larger `llama3.1:8b` model and a shorter recording window (3 seconds).

```bash
# Terminal 1 — Start Ollama
ollama serve

# Terminal 2 — Start drone control
python drone_control_model_c.py
```

Everything else works the same as Model B. Expect higher command extraction accuracy but also higher latency compared to Model B.

---

### inference.py — Test without Drone

A standalone tool to test the CNN model without connecting to a drone. Shows the top-3 predictions with confidence scores and a visual bar chart in the terminal.

```bash
python inference.py
```

This is useful for verifying that Model A works correctly before flying.

---

## Commands Reference

### Model A

| Keyword | Action | Distance/Angle |
|---------|--------|----------------|
| `forward` | Move forward | 100 cm |
| `backward` | Move backward | 100 cm |
| `up` | Move up | 30 cm |
| `down` | Move down | 30 cm |
| `left` | Move left | 30 cm |
| `right` | Move right | 30 cm |
| `go` | Take off | — |
| `stop` | Land | — |
| `one` | Rotate clockwise | 45° |
| `two` | Rotate counter-clockwise | 45° |

### Model B & C

| Command | Action | Default Parameter |
|---------|--------|-------------------|
| `forward` | Move forward | 30 cm |
| `backward` | Move backward | 30 cm |
| `up` | Move up | 30 cm |
| `down` | Move down | 30 cm |
| `left` | Move left | 30 cm |
| `right` | Move right | 30 cm |
| `rotate left` | Rotate counter-clockwise | 30° |
| `rotate right` | Rotate clockwise | 30° |
| `start` | Take off | — |
| `stop` | Land | — |

Parameters are extracted from natural language. Example: *"go forward 80 centimeters"* → `forward`, 80.

---

## Logging & Latency Measurement

All three drone control modules automatically log every session and measure latency for each pipeline phase.

### Log Files

A new timestamped log file is created for every program start:

```
logs/model_a_20250519_143012.log
logs/model_b_20250519_151203.log
```

Each log entry includes a timestamp and covers all console output — model loading, transcriptions, LLM responses, drone commands, and errors.

Example log output (Model B):
```
2025-05-19 14:32:01  ● Aufnahme läuft...
2025-05-19 14:32:06  Transkribiere...
2025-05-19 14:32:07  Erkannter Text: fly forward 50 centimeters
2025-05-19 14:32:07  Kommando Extrahierung...
2025-05-19 14:32:09  LLaMA Antwort: {"command": "forward", "parameter": 50}
2025-05-19 14:32:09  Latenz → Aufnahme: 5.002s | Whisper: 1.243s | LLM: 2.187s | Gesamt: 8.432s
```

### Latency CSV Files

A cumulative CSV file grows across all sessions:

```
logs/latencies_model_a.csv
logs/latencies_model_b.csv
logs/latencies_model_c.csv
```

**Model A columns:** `timestamp`, `command`, `confidence`, `recording_s`, `preprocessing_s`, `inference_s`, `total_s`

**Model B & C columns:** `timestamp`, `command`, `parameter`, `recording_s`, `whisper_s`, `llm_s`, `total_s`

Load and compare with pandas:
```python
import pandas as pd

df_a = pd.read_csv('logs/latencies_model_a.csv')
df_b = pd.read_csv('logs/latencies_model_b.csv')
df_c = pd.read_csv('logs/latencies_model_c.csv')

print(df_a['total_s'].describe())
print(df_b['total_s'].describe())
print(df_c['total_s'].describe())
```

---

## Notebooks

Run notebooks in this order:

1. **`notebooks/data_exploration.ipynb`** — Explore the dataset: class distribution, audio duration statistics, sample waveforms and spectrograms
2. **`notebooks/preprocessing.ipynb`** — Data augmentation (pitch shift, time shift) for underrepresented classes, dataset splitting into train/val/test
3. **`notebooks/model_a.ipynb`** — CNN model definition, training, evaluation, confusion matrix, training curves

---

## Results

Training results for Model A are stored in `results/`:

- `model_a_confusion_matrix.png` — Confusion matrix across all 12 classes
- `model_a_training_curves.png` — Training and validation accuracy/loss over epochs

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
