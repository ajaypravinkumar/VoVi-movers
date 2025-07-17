from vosk import Model, KaldiRecognizer
import pyaudio
import json
import serial
from time import sleep

# --------- Serial Communication Setup ---------
try:
    ser = serial.Serial("COM5", baudrate=9600, timeout=0.5)
    print("Serial port COM5 connected.")
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    ser = None

# --------- Allowed Commands & Serial Mappings ---------
command_serial_map = {
    "forward": ("1", "Moving forward..."),
    "backward": ("2", "Moving backward..."),
    "right": ("3", "Turning right..."),
    "left": ("4", "Turning left..."),
    "stop": ("5", "Stopping wheelchair..."),
    "break": ("5", "Stopping wheelchair..."),  
}

allowed_commands = list(command_serial_map.keys())

# --------- Vosk Model Setup ---------
model = Model("vosk-model-small-en-in-0.4")
recognizer = KaldiRecognizer(model, 16000, json.dumps(allowed_commands))

# --------- Microphone Setup ---------
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
stream.start_stream()

# --------- Function to Send Serial Command ---------
def send_serial_command(command):
    if command in command_serial_map:
        code, msg = command_serial_map[command]
        print(f"Voice command recognized: {command} --> Serial command: {code}")
        if ser:
            ser.write(code.encode())
        print(f"Action: {msg}")
    else:
        print(f"Ignored unrecognized command: {command}")

# --------- Main Listening Loop ---------
print("Listening for voice commands (offline)...")

try:
    while True:
        data = stream.read(4096, exception_on_overflow=False)
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            command = result.get("text", "").lower().strip()
            if command:
                send_serial_command(command)
except KeyboardInterrupt:
    print("\nExiting gracefully...")
    stream.stop_stream()
    stream.close()
    p.terminate()
    if ser:
        ser.close()
