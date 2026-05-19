"""
UAV VoiceControl — Echtzeit-Inferenz
=====================================
Aufnahme per Mikrofon → Mel-Spektrogramm → CNN-Klassifizierung

Starten:
    python inference.py

Steuerung:
    Enter   → 1 Sekunde aufnehmen & klassifizieren
    q       → Beenden
"""

import sys
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
from pathlib import Path

# ── Konfiguration ──────────────────────────────────────────────────────────
BASE_PATH   = Path(__file__).parent
MODEL_PATH  = BASE_PATH / 'models' / 'best_model.keras'

SAMPLE_RATE   = 16000
N_MELS        = 64
N_FFT         = 400
HOP_LENGTH    = 160
INPUT_SHAPE   = (98, 64, 1)

LABELS = [
    'forward', 'backward', 'up', 'down', 'stop',
    'right',   'left',     'go', 'one',  'two',
    'silence', 'unknown'
]

# Farbcodes für Terminal-Output
GREEN  = '\033[92m'
YELLOW = '\033[93m'
RED    = '\033[91m'
BLUE   = '\033[94m'
RESET  = '\033[0m'
BOLD   = '\033[1m'


# ── Preprocessing ──────────────────────────────────────────────────────────
def center_pad(audio: np.ndarray) -> np.ndarray:
    if len(audio) >= SAMPLE_RATE:
        return audio[:SAMPLE_RATE]
    pad_total = SAMPLE_RATE - len(audio)
    pad_left  = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(audio, (pad_left, pad_right))


def compute_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmax=8000, center=False
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).T          # (98, 64)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    return log_mel[np.newaxis, ..., np.newaxis]                # (1, 98, 64, 1)


# ── Aufnahme ───────────────────────────────────────────────────────────────
def record(duration: float = 1.0) -> np.ndarray:
    print(f'{BLUE}  ● Aufnahme läuft...{RESET}', end='', flush=True)
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    print(f'\r{BLUE}  ● Aufnahme fertig.  {RESET}')
    return audio.flatten()


# ── Inferenz ───────────────────────────────────────────────────────────────
def predict(model, audio: np.ndarray):
    audio  = center_pad(audio)
    spec   = compute_mel_spectrogram(audio)
    probs  = model.predict(spec, verbose=0)[0]
    top3   = np.argsort(probs)[::-1][:3]
    return probs, top3


def format_result(probs, top3):
    best_label = LABELS[top3[0]]
    best_conf  = probs[top3[0]] * 100

    # Farbe je nach Confidence
    if best_conf >= 80:
        color = GREEN
    elif best_conf >= 50:
        color = YELLOW
    else:
        color = RED

    # Ergebnis-Box
    print()
    print('  ┌─────────────────────────────────────┐')
    print(f'  │  {BOLD}Erkannt:{RESET}  {color}{BOLD}{best_label.upper():<12}{RESET}  ({best_conf:5.1f}%)     │')
    print('  ├─────────────────────────────────────┤')
    print('  │  Top 3:                             │')
    for rank, idx in enumerate(top3):
        bar_len = int(probs[idx] * 20)
        bar     = '█' * bar_len + '░' * (20 - bar_len)
        print(f'  │  {rank+1}. {LABELS[idx]:<10} {bar} {probs[idx]*100:5.1f}%  │')
    print('  └─────────────────────────────────────┘')
    print()


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print()
    print(f'{BOLD}╔══════════════════════════════════════════╗{RESET}')
    print(f'{BOLD}║    UAV VoiceControl — Echtzeit-Demo      ║{RESET}')
    print(f'{BOLD}╚══════════════════════════════════════════╝{RESET}')
    print()

    # Modell laden
    print('  Lade Modell...', end='', flush=True)
    if not MODEL_PATH.exists():
        print(f'\n{RED}  Fehler: Modell nicht gefunden: {MODEL_PATH}{RESET}')
        sys.exit(1)
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f' {GREEN}✓{RESET}')

    print(f'  Klassen: {", ".join(LABELS)}')
    print()
    print(f'  {BOLD}Enter{RESET} → aufnehmen & klassifizieren')
    print(f'  {BOLD}q{RESET}     → beenden')
    print()

    while True:
        try:
            user_input = input('  Warte... (Enter zum Sprechen) → ').strip().lower()
        except (KeyboardInterrupt, EOFError):
            break

        if user_input == 'q':
            break

        audio        = record()
        probs, top3  = predict(model, audio)
        format_result(probs, top3)

    print(f'\n{YELLOW}  Beendet.{RESET}\n')


if __name__ == '__main__':
    main()
