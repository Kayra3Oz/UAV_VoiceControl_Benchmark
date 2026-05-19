# UAV VoiceControl Benchmark

> **[English Version](README.md)**

Ein Benchmark-Projekt, das drei verschiedene sprachgesteuerte UAV-Pipelines auf einer **DJI Tello** Drohne vergleicht. Das Projekt untersucht den Trade-off zwischen Latenz, Genauigkeit und Systemkomplexität bei drei grundlegend unterschiedlichen Ansätzen: einem leichtgewichtigen CNN-Klassifizierer sowie zwei LLM-basierten Pipelines mit Whisper zur Spracherkennung und LLaMA zur Befehlsextraktion.

---

## Inhaltsverzeichnis

- [Projektübersicht](#projektübersicht)
- [Architektur](#architektur)
- [Datensatz](#datensatz)
- [Projektstruktur](#projektstruktur)
- [Voraussetzungen](#voraussetzungen)
- [Installation](#installation)
- [Bedienung](#bedienung)
  - [Model A — CNN (Keyword Spotting)](#model-a--cnn-keyword-spotting)
  - [Model B — Whisper + LLaMA 3.2:3b](#model-b--whisper--llama-32-3b)
  - [Model C — Whisper + LLaMA 3.1:8b](#model-c--whisper--llama-31-8b)
  - [inference.py — Test ohne Drohne](#inferencepy--test-ohne-drohne)
- [Befehlsreferenz](#befehlsreferenz)
- [Logging & Latenzmessung](#logging--latenzmessung)
- [Notebooks](#notebooks)
- [Ergebnisse](#ergebnisse)

---

## Projektübersicht

Dieses Projekt vergleicht drei Sprachsteuerungs-Pipelines für die UAV-Navigation:

| | Model A | Model B | Model C |
|---|---|---|---|
| **Ansatz** | CNN Keyword Spotting | Whisper + LLaMA 3.2:3b | Whisper + LLaMA 3.1:8b |
| **Aufnahmedauer** | 1 Sekunde | 5 Sekunden | 3 Sekunden |
| **Spracherkennung** | Mel-Spektrogramm → CNN | OpenAI Whisper (base) | OpenAI Whisper (base) |
| **Befehlsextraktion** | Direkte Klassifikation | LLaMA via Ollama API | LLaMA via Ollama API |
| **Parameter möglich** | Nein (feste Abstände) | Ja (z.B. „vorwärts 80 cm") | Ja (z.B. „vorwärts 80 cm") |
| **Internet erforderlich** | Nein | Nein (läuft lokal) | Nein (läuft lokal) |

---

## Architektur

### Model A — CNN

Die CNN-Architektur basiert auf dem Stanford CS229 Paper von Li & Zhou (2017), angepasst für dieses Projekt:

- **Eingabe:** Log-Mel-Spektrogramm der Form `(98, 64, 1)` — berechnet aus 1 Sekunde Audio bei 16 kHz
- **Architektur:** 3 Convolutional Layer mit BatchNormalization, MaxPooling und Dropout, gefolgt von einem Dense-Output-Layer
- **Ausgabe:** 12-Klassen Softmax — 10 UAV-Keywords + `silence` + `unknown`
- **Trainingsdaten:** Google Speech Commands Dataset mit Augmentierung für unterrepräsentierte Klassen

**Preprocessing-Pipeline:**
1. 1 Sekunde Audio bei 16 kHz aufnehmen
2. Center-Padding oder Trimmen auf exakt 16.000 Samples
3. Log-Mel-Spektrogramm berechnen (n_mels=64, n_fft=400, hop_length=160, fmax=8000)
4. Per-Sample-Normalisierung (Mittelwert=0, Std=1)
5. CNN-Inferenz → vorhergesagtes Label + Konfidenz

### Model B & C — Whisper + LLaMA

Diese Pipelines verwenden einen zweistufigen Ansatz:

1. **Whisper (base):** Transkribiert das aufgenommene Audio auf Englisch
2. **LLaMA via Ollama:** Erhält die Transkription und extrahiert daraus einen strukturierten Drohnenbefehl (JSON) aus natürlicher Sprache

Das ermöglicht freie Sprachbefehle wie *„please fly forward fifty centimeters"*, die das LLM auf `{"command": "forward", "parameter": 50}` abbildet.

Model B verwendet `llama3.2:3b` (schneller, geringerer Ressourcenbedarf), Model C verwendet `llama3.1:8b` (leistungsfähiger, höhere Latenz).

---

## Datensatz

Dieses Projekt verwendet das **Google Speech Commands Dataset v2**.

**Download:** [https://www.tensorflow.org/datasets/catalog/speech_commands](https://www.tensorflow.org/datasets/catalog/speech_commands)

Direkter Download-Link:
```
http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
```

Nach dem Herunterladen den Datensatz entpacken und unter folgendem Pfad ablegen:
```
data/raw/speech_command_data/
```

Das Verzeichnis sollte einen Ordner pro Wort enthalten, z.B.:
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

**Verwendete Klassen (12 gesamt):**
`forward`, `backward`, `up`, `down`, `stop`, `right`, `left`, `go`, `one`, `two`, `silence`, `unknown`

**Preprocessing-Schritte** (siehe `notebooks/preprocessing.ipynb`):
- **Augmentierung** der unterrepräsentierten Klassen `forward` und `backward` mittels Pitch Shift (±2 Halbtöne) und Time Shift (±10%)
- **Downsampling** der `unknown`-Klasse zur Balancierung des Datensatzes
- Alle Clips werden center-gepaddet oder auf exakt **1 Sekunde (16.000 Samples)** getrimmt

---

## Projektstruktur

```
UAV_VoiceControl_Benchmark/
│
├── drone_control_model_a.py    # CNN-basierte Drohnensteuerung
├── drone_control_model_b.py    # Whisper + LLaMA 3.2:3b Drohnensteuerung
├── drone_control_model_c.py    # Whisper + LLaMA 3.1:8b Drohnensteuerung
├── inference.py                # Eigenständiges CNN-Testtool (ohne Drohne)
├── requirements.txt            # Python-Abhängigkeiten
│
├── notebooks/
│   ├── data_exploration.ipynb  # Datensatzanalyse und Visualisierung
│   ├── preprocessing.ipynb     # Datenaugmentierung und Split-Erstellung
│   └── model_a.ipynb           # CNN-Training und Evaluation
│
├── models/
│   └── model_a.keras           # Trainiertes CNN-Modell (nicht im Repository)
│
├── data/                       # Nicht im Repository — siehe Datensatz-Abschnitt
│   ├── raw/
│   ├── augmented/
│   ├── spectrograms/
│   └── splits/
│
├── results/
│   ├── model_a_confusion_matrix.png
│   └── model_a_training_curves.png
│
└── logs/                       # Wird automatisch zur Laufzeit erstellt
    ├── model_a_JJJJMMTT_HHMMSS.log
    ├── model_b_JJJJMMTT_HHMMSS.log
    ├── model_c_JJJJMMTT_HHMMSS.log
    ├── latencies_model_a.csv
    ├── latencies_model_b.csv
    └── latencies_model_c.csv
```

---

## Voraussetzungen

### Python-Abhängigkeiten

Installation mit:
```bash
pip install -r requirements.txt
```

Inhalt der `requirements.txt`:
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

### Ollama (nur für Model B und C)

Ollama ist kein Python-Paket — es ist eine eigenständige Anwendung, die LLMs lokal ausführt.

1. Ollama herunterladen und installieren: [https://ollama.com](https://ollama.com)
2. Benötigte Modelle herunterladen:

```bash
# Für Model B
ollama pull llama3.2:3b

# Für Model C
ollama pull llama3.1:8b
```

3. Ollama vor dem Start von Model B oder C starten:
```bash
ollama serve
```

### Hardware

- **DJI Tello** Drohne (erforderlich für die Drohnensteuerungs-Module)
- Mikrofon (eingebaut oder extern)
- Python 3.9 oder höher empfohlen

---

## Installation

```bash
# 1. Repository klonen
git clone https://github.com/Kayra3Oz/UAV_VoiceControl_Benchmark.git
cd UAV_VoiceControl_Benchmark

# 2. Virtuelle Umgebung erstellen und aktivieren
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Python-Abhängigkeiten installieren
pip install -r requirements.txt

# 4. Datensatz herunterladen (siehe Datensatz-Abschnitt oben)
#    Ablegen unter: data/raw/speech_command_data/

# 5. Preprocessing-Notebooks ausführen (in dieser Reihenfolge)
#    notebooks/preprocessing.ipynb
#    notebooks/model_a.ipynb

# 6. (Optional) Ollama für Model B / C installieren
#    https://ollama.com
#    ollama pull llama3.2:3b
#    ollama pull llama3.1:8b
```

---

## Bedienung

> **Wichtig:** Für alle Drohnensteuerungs-Module vor dem Start mit dem **WLAN-Netz der DJI Tello** verbinden.

---

### Model A — CNN (Keyword Spotting)

Die schnellste Pipeline. Ein einzelnes Keyword sprechen — das CNN klassifiziert es direkt.

```bash
python drone_control_model_a.py
```

**Ablauf:**
1. Das Skript lädt das trainierte CNN-Modell (`models/model_a.keras`)
2. **Enter** drücken, um eine 1-Sekunden-Aufnahme zu starten
3. Das Audio wird als Mel-Spektrogramm vorverarbeitet und dem CNN übergeben
4. Der vorhergesagte Befehl wird an die Drohne gesendet
5. **q + Enter** zum Beenden

**Unterstützte Befehle (genau diese Wörter sprechen):**

| Wort | Drohnenaktion |
|------|--------------|
| `forward` | 100 cm vorwärts fliegen |
| `backward` | 100 cm rückwärts fliegen |
| `up` | 30 cm nach oben fliegen |
| `down` | 30 cm nach unten fliegen |
| `left` | 30 cm nach links fliegen |
| `right` | 30 cm nach rechts fliegen |
| `go` | Abheben |
| `stop` | Landen |
| `one` | 45° im Uhrzeigersinn drehen |
| `two` | 45° gegen den Uhrzeigersinn drehen |

---

### Model B — Whisper + LLaMA 3.2:3b

Natürlichsprachliche Pipeline. Ganze Sätze sprechen — das LLM extrahiert den Befehl.

**Voraussetzung:** Ollama muss laufen und `llama3.2:3b` muss heruntergeladen sein.

```bash
# Terminal 1 — Ollama starten
ollama serve

# Terminal 2 — Drohnensteuerung starten
python drone_control_model_b.py
```

**Ablauf:**
1. **Enter** drücken, um eine 5-Sekunden-Aufnahme zu starten
2. Whisper transkribiert die Sprache in Text
3. LLaMA 3.2:3b extrahiert einen strukturierten Befehl aus der Transkription
4. Befehl + Parameter werden an die Drohne gesendet
5. **q + Enter** zum Beenden

**Beispielbefehle:**

| Gesprochener Satz | Extrahierter Befehl | Parameter |
|-------------------|---------------------|-----------|
| "fly forward fifty centimeters" | `forward` | 50 |
| "go up thirty" | `up` | 30 |
| "rotate to the left" | `rotate left` | 30 |
| "stop" | `stop` | 0 |
| "take off" | `start` | 0 |

**Unterstützte Befehle:** `forward`, `backward`, `right`, `left`, `up`, `down`, `start`, `stop`, `rotate left`, `rotate right`

Wird kein Parameter angegeben, wird standardmäßig **30 cm** verwendet.

---

### Model C — Whisper + LLaMA 3.1:8b

Identische Pipeline wie Model B, jedoch mit dem größeren `llama3.1:8b` Modell und einem kürzeren Aufnahmefenster (3 Sekunden).

```bash
# Terminal 1 — Ollama starten
ollama serve

# Terminal 2 — Drohnensteuerung starten
python drone_control_model_c.py
```

Alles weitere funktioniert identisch zu Model B. Im Vergleich zu Model B ist eine höhere Befehlsextraktionsgenauigkeit zu erwarten, jedoch auch eine höhere Latenz.

---

### inference.py — Test ohne Drohne

Ein eigenständiges Testtool für das CNN-Modell ohne Drohnenverbindung. Zeigt die Top-3-Vorhersagen mit Konfidenzwerten und einem visuellen Balkendiagramm direkt im Terminal an.

```bash
python inference.py
```

Nützlich zum Überprüfen, ob Model A korrekt funktioniert, bevor die Drohne gestartet wird.

---

## Befehlsreferenz

### Model A

| Keyword | Aktion | Distanz/Winkel |
|---------|--------|----------------|
| `forward` | Vorwärts bewegen | 100 cm |
| `backward` | Rückwärts bewegen | 100 cm |
| `up` | Nach oben bewegen | 30 cm |
| `down` | Nach unten bewegen | 30 cm |
| `left` | Nach links bewegen | 30 cm |
| `right` | Nach rechts bewegen | 30 cm |
| `go` | Abheben | — |
| `stop` | Landen | — |
| `one` | Im Uhrzeigersinn drehen | 45° |
| `two` | Gegen den Uhrzeigersinn drehen | 45° |

### Model B & C

| Befehl | Aktion | Standard-Parameter |
|--------|--------|-------------------|
| `forward` | Vorwärts bewegen | 30 cm |
| `backward` | Rückwärts bewegen | 30 cm |
| `up` | Nach oben bewegen | 30 cm |
| `down` | Nach unten bewegen | 30 cm |
| `left` | Nach links bewegen | 30 cm |
| `right` | Nach rechts bewegen | 30 cm |
| `rotate left` | Gegen den Uhrzeigersinn drehen | 30° |
| `rotate right` | Im Uhrzeigersinn drehen | 30° |
| `start` | Abheben | — |
| `stop` | Landen | — |

Parameter werden aus natürlicher Sprache extrahiert. Beispiel: *„go forward 80 centimeters"* → `forward`, 80.

---

## Logging & Latenzmessung

Alle drei Drohnensteuerungs-Module protokollieren jede Session automatisch und messen die Latenz für jede Pipeline-Phase.

### Log-Dateien

Bei jedem Programmstart wird eine neue, mit Zeitstempel versehene Log-Datei erstellt:

```
logs/model_a_20250519_143012.log
logs/model_b_20250519_151203.log
```

Jeder Log-Eintrag enthält einen Zeitstempel und erfasst alle Konsolenausgaben — Modell-Loading, Transkriptionen, LLM-Antworten, Drohnenbefehle und Fehler.

Beispiel-Log-Ausgabe (Model B):
```
2025-05-19 14:32:01  ● Aufnahme läuft...
2025-05-19 14:32:06  Transkribiere...
2025-05-19 14:32:07  Erkannter Text: fly forward 50 centimeters
2025-05-19 14:32:07  Kommando Extrahierung...
2025-05-19 14:32:09  LLaMA Antwort: {"command": "forward", "parameter": 50}
2025-05-19 14:32:09  Latenz → Aufnahme: 5.002s | Whisper: 1.243s | LLM: 2.187s | Gesamt: 8.432s
```

### Latenz-CSV-Dateien

Eine kumulative CSV-Datei wächst über alle Sessions hinweg:

```
logs/latencies_model_a.csv
logs/latencies_model_b.csv
logs/latencies_model_c.csv
```

**Model A Spalten:** `timestamp`, `command`, `confidence`, `recording_s`, `preprocessing_s`, `inference_s`, `total_s`

**Model B & C Spalten:** `timestamp`, `command`, `parameter`, `recording_s`, `whisper_s`, `llm_s`, `total_s`

Mit pandas laden und vergleichen:
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

Notebooks in dieser Reihenfolge ausführen:

1. **`notebooks/data_exploration.ipynb`** — Datensatz erkunden: Klassenverteilung, Audio-Dauer-Statistiken, Beispiel-Wellenformen und Spektrogramme
2. **`notebooks/preprocessing.ipynb`** — Datenaugmentierung (Pitch Shift, Time Shift) für unterrepräsentierte Klassen, Datensatz-Splits in Train/Val/Test
3. **`notebooks/model_a.ipynb`** — CNN-Modelldefinition, Training, Evaluation, Konfusionsmatrix, Trainingskurven

---

## Ergebnisse

Trainingsergebnisse für Model A sind unter `results/` gespeichert:

- `model_a_confusion_matrix.png` — Konfusionsmatrix über alle 12 Klassen
- `model_a_training_curves.png` — Trainings- und Validierungsgenauigkeit/-verlust über die Epochen

---

## Lizenz

Dieses Projekt steht unter der **MIT-Lizenz**. Details siehe [LICENSE](LICENSE).
