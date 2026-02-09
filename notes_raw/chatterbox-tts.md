TTS Server (chatterbox)

# Create huggingface cache on NVME
mkdir -p /home/eric/projects/huggingface/hub
echo 'export HF_HOME=/home/eric/projects/huggingface/hub' >> ~/.bashrc
source ~/.bashrc

# Create venv and install dependencies
We need torch 2.8.0 as the default torch thats downloaded isn't cuda enabled

python3 -m venv .venv
pip3 install chatterbox-tts
python3 -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126

IGNORE THIS: ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
chatterbox-tts 0.1.6 requires torch==2.6.0, but you have torch 2.8.0 which is incompatible.
chatterbox-tts 0.1.6 requires torchaudio==2.6.0, but you have torchaudio 2.8.0 which is incompatible.

Quick test script for multilingual:
# Test script (Dutch)
``` python3
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Load the multilingual model once
multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

# Generate and save Dutch speech
dutch_texts = [
    "Goedemorgen! Hoe gaat het met je, dit is chatterbox in het Nederlands!",
    "Het is fris buiten, ik ga even een kop koffie pakken en een dikke trui aan doen."
]

for i, text in enumerate(dutch_texts, start=1):
    wav_nl = multilingual_model.generate(text, language_id="nl")
    ta.save(f"test{i}-dutch.wav", wav_nl, multilingual_model.sr)
    print(f"Saved test{i}-dutch.wav")
```
