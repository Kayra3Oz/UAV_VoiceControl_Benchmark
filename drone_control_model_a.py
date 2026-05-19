"""
UAV VoiceControl — Drohnensteuerung
=====================================
Enter drücken → 1 Sekunde aufnehmen → Prediction als String zurück

Starten:
    python drone_control.py

Steuerung:
    Enter  → 1 Sekunde aufnehmen & klassifizieren
    q + Enter  → Beenden
"""

import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
from pathlib import Path
from djitellopy import Tello
import logging
import time
import csv
from datetime import datetime

# ── Logging Setup ──────────────────────────────────────────────────────────
BASE_PATH  = Path(__file__).parent
LOG_DIR    = BASE_PATH / 'logs'
LOG_DIR.mkdir(exist_ok=True)

_session_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
_log_file   = LOG_DIR / f'model_a_{_session_ts}.log'
_csv_file   = LOG_DIR / 'latencies_model_a.csv'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(_log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# CSV-Header nur beim allerersten Mal schreiben
if not _csv_file.exists():
    with open(_csv_file, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([
            'timestamp', 'command', 'confidence',
            'recording_s', 'preprocessing_s', 'inference_s', 'total_s'
        ])

# ── Konfiguration ──────────────────────────────────────────────────────────
MODEL_A_PATH = BASE_PATH / 'models' / 'model_a.keras'

SAMPLE_RATE = 16000
N_MELS      = 64
N_FFT       = 400
HOP_LENGTH  = 160

LABELS = [
    'forward', 'backward', 'up', 'down', 'stop',
    'right',   'left',     'go', 'one',  'two',
    'silence', 'unknown'
]


# ── Preprocessing ──────────────────────────────
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
    log_mel = librosa.power_to_db(mel, ref=np.max).T           # (98, 64)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    return log_mel[np.newaxis, ..., np.newaxis]                 # (1, 98, 64, 1)


# ── Aufnahme ───────────────────────────────────────────────────────────────
def record_1s() -> np.ndarray:
    audio = sd.rec(
        int(1.0 * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    return audio.flatten()


# ── Kernfunktion: gibt Prediction + Latenzen zurück ───────────────────────
def get_prediction(model_a) -> tuple:
    # Phase 1: Aufnahme
    t0    = time.perf_counter()
    audio = record_1s()
    t1    = time.perf_counter()

    # Phase 2: Preprocessing
    audio = center_pad(audio)
    spec  = compute_mel_spectrogram(audio)
    t2    = time.perf_counter()

    # Phase 3: CNN-Inferenz
    probs      = model_a.predict(spec, verbose=0)[0]   # Array mit 12 Werten
    t3         = time.perf_counter()

    best_index = np.argmax(probs)                      # Index der höchsten Prob.
    prediction = LABELS[best_index]                    # String, z.B. "forward"
    confidence = probs[best_index]                     # z.B. 0.92

    latencies = {
        'recording_s':     round(t1 - t0, 4),
        'preprocessing_s': round(t2 - t1, 4),
        'inference_s':     round(t3 - t2, 4),
        'total_s':         round(t3 - t0, 4),
    }

    return prediction, confidence, latencies


drone = Tello()


# ── Hier kannst du die Drohne steuern ─────────────────────────────────────
def handle_command(command: str, confidence: float):
    log.info(f'→ Befehl erkannt: "{command}"  ({confidence*100:.1f}%)')

    if command == 'forward':
        drone.move_forward(100)
    elif command == 'backward':
        drone.move_back(100)
    elif command == 'up':
        drone.move_up(30)
    elif command == 'down':
        drone.move_down(30)
    elif command == 'left':
        drone.move_left(30)
    elif command == 'right':
        drone.move_right(30)
    elif command == 'stop':
        drone.land()
    elif command == 'go':
        drone.takeoff()
    elif command == 'one':
        drone.rotate_clockwise(45)
    elif command == 'two':
        drone.rotate_counter_clockwise(45)
    elif command in ('silence', 'unknown'):
        log.info('→ Kein gültiger Befehl erkannt.')


def connect_to_drone():
    drone.connect()

def handle_battery_power(power):
    if power < 11:
        log.info(f'Battery Power: {power}! Landing...')
        drone.land()


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    log.info('Lade Modell...')
    model_a = tf.keras.models.load_model(MODEL_A_PATH)
    log.info('Modell A (CNN) geladen!')
    log.info('Enter      → sprechen (1 Sekunde)')
    log.info('q + Enter  → beenden')

    while True:
        user_input = input('  Warte... (Enter zum Sprechen, q zum Beenden) → ').strip().lower()
        if user_input == 'q':
            break

        log.info('● Aufnahme läuft...')
        command, confidence, lat = get_prediction(model_a)

        log.info(
            f'Latenz → Aufnahme: {lat["recording_s"]:.3f}s | '
            f'Preprocessing: {lat["preprocessing_s"]:.3f}s | '
            f'Inferenz: {lat["inference_s"]:.3f}s | '
            f'Gesamt: {lat["total_s"]:.3f}s'
        )

        # Latenzen in CSV speichern (append)
        with open(_csv_file, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(),
                command,
                f'{confidence:.4f}',
                lat['recording_s'],
                lat['preprocessing_s'],
                lat['inference_s'],
                lat['total_s'],
            ])

        handle_command(command, confidence)


if __name__ == '__main__':
    connect_to_drone()
    main()
