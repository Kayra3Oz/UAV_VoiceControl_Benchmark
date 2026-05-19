import numpy as np
import sounddevice as sd
from pathlib import Path
import requests
import json
import re
from djitellopy import Tello
import whisper
import logging
import time
import csv
from datetime import datetime

# ── Logging Setup ──────────────────────────────────────────────────────────
BASE_PATH = Path(__file__).parent
LOG_DIR   = BASE_PATH / 'logs'
LOG_DIR.mkdir(exist_ok=True)

_session_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
_log_file   = LOG_DIR / f'model_c_{_session_ts}.log'
_csv_file   = LOG_DIR / 'latencies_model_c.csv'

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
            'timestamp', 'command', 'parameter',
            'recording_s', 'whisper_s', 'llm_s', 'total_s'
        ])

# ── Konfiguration ──────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
DURATION = 3.0

OLLAMA_URL = 'http://localhost:11434/api/generate'
OLLAMA_MODEL = 'llama3.1:8b'

VALID_COMMANDS = {'forward', 'backward', 'right', 'left', 'up', 'down', 'start', 'stop', 'rotate left', 'rotate right'}

drone = Tello()


def record_1s() -> np.ndarray:
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    return audio.flatten()


def speech_to_text(model, audio: np.ndarray) -> str:
    transcription = model.transcribe(audio, language='en')
    return transcription['text'].strip()


def extract_command(text: str) -> dict:
    prompt = f'''Extract a drone command from this sentence: "{text}"

Valid commands (EXACTLY as written, nothing else): forward, backward, right, left, up, down, start, stop, rotate left, rotate right
Rules:
- The command must be EXACTLY one of the valid commands above. Words like "fly", "please", "move" are NOT commands.
- "go" and "stop" always have parameter 0
- All other commands need a number parameter in cm. If no number is given, use 30.
- If no valid command found, use "unknown command" and parameter 0

Return ONLY this JSON, nothing else as an example:
{{"command": "up", "parameter": 100}}

JSON:'''

    response = requests.post(
        OLLAMA_URL,
        json={
            'model': OLLAMA_MODEL,
            'prompt': prompt,
            'stream': False
        })

    raw = response.json()['response'].strip()
    log.info(f'LLaMA Antwort: "{raw}"')

    match = re.search(r'\{.*?\}', raw, re.DOTALL)
    if match:
        result = json.loads(match.group())
        # Validierung: command muss exakt in der gültigen Liste sein
        if result.get('command') not in VALID_COMMANDS:
            log.info(f'→ Ungültiger Command "{result.get("command")}", wird abgewiesen.')
            return {"command": "unknown command", "parameter": 0}
        return result
    else:
        return {"command": "unknown command", "parameter": 0}


def handle_dron(command_param: dict):
    command = command_param['command']
    parameter = command_param['parameter']

    if command == 'unknown command':
        log.info('→ Unknown command')

    elif command == 'forward':
        if parameter == 0:
            log.info('→ Parameter cannot be 0')
            return
        log.info(f'Command: {command}, Parameter: {parameter}')
        drone.move_forward(parameter)

    elif command == 'backward':
        if parameter == 0:
            log.info('→ Parameter cannot be 0')
            return
        log.info(f'Command: {command}, Parameter: {parameter}')
        drone.move_back(parameter)

    elif command == 'right':
        if parameter == 0:
            log.info('→ Parameter cannot be 0')
            return
        log.info(f'Command: {command}, Parameter: {parameter}')
        drone.move_right(parameter)

    elif command == 'left':
        if parameter == 0:
            log.info('→ Parameter cannot be 0')
            return
        log.info(f'Command: {command}, Parameter: {parameter}')
        drone.move_left(parameter)

    elif command == 'up':
        if parameter == 0:
            log.info('→ Parameter cannot be 0')
            return
        log.info(f'Command: {command}, Parameter: {parameter}')
        drone.move_up(parameter)

    elif command == 'down':
        if parameter == 0:
            log.info('→ Parameter cannot be 0')
            return
        log.info(f'Command: {command}, Parameter: {parameter}')
        drone.move_down(parameter)

    elif command == 'start':
        log.info(f'Command: {command}')
        drone.takeoff()

    elif command == 'stop':
        log.info(f'Command: {command}')
        drone.land()

    elif command == 'rotate left':
        if parameter == 0:
            log.info('→ Parameter cannot be 0')
            return
        log.info(f'Command: {command}, Parameter: {parameter}')
        drone.rotate_counter_clockwise(parameter)

    elif command == 'rotate right':
        if parameter == 0:
            log.info('→ Parameter cannot be 0')
            return
        log.info(f'Command: {command}, Parameter: {parameter}')
        drone.rotate_clockwise(parameter)


def connect_to_drone():
    drone.connect()


if __name__ == '__main__':

    log.info('Whisper model laden...')
    model = whisper.load_model('base')
    log.info('Whisper geladen!')

    connect_to_drone()

    log.info('Enter -> sprechen (3 Sekunden)')
    log.info('q + Enter -> Programm beenden')

    while True:
        log.info(f'Akkustand: {drone.get_battery()}')
        user_input = input('Bereit...(Enter zum sprechen, q zum beenden) -> ').strip().lower()
        if user_input == 'q':
            drone.land()
            break

        # Phase 1: Aufnahme
        log.info('● Aufnahme läuft...')
        t0    = time.perf_counter()
        audio = record_1s()
        t1    = time.perf_counter()

        # Phase 2: Whisper Transkription
        log.info('Transkribiere...')
        text = speech_to_text(model, audio)
        t2   = time.perf_counter()
        log.info(f'Erkannter Text: {text}')

        # Phase 3: LLM Kommando-Extraktion
        log.info('Kommando Extrahierung...')
        command_dict = extract_command(text)
        t3           = time.perf_counter()
        log.info(f'Extraktionsergebnisse: command: {command_dict["command"]}, parameter: {command_dict["parameter"]}')

        # Latenzen berechnen und loggen
        lat_rec = round(t1 - t0, 4)
        lat_wsp = round(t2 - t1, 4)
        lat_llm = round(t3 - t2, 4)
        lat_tot = round(t3 - t0, 4)

        log.info(
            f'Latenz → Aufnahme: {lat_rec:.3f}s | '
            f'Whisper: {lat_wsp:.3f}s | '
            f'LLM: {lat_llm:.3f}s | '
            f'Gesamt: {lat_tot:.3f}s'
        )

        # Latenzen in CSV speichern (append)
        with open(_csv_file, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(),
                command_dict['command'],
                command_dict['parameter'],
                lat_rec, lat_wsp, lat_llm, lat_tot,
            ])

        try:
            handle_dron(command_dict)
        except Exception as e:
            log.info(f'→ Drohnenfehler: {e}')
