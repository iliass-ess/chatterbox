import os
from fastapi import FastAPI
from pathlib import Path
import torchaudio as ta

from chatterbox.tts_extended import ChatterboxTTS

app = FastAPI()

RAM_DIR = Path("/dev/shm/tts_pipeline")
DEVICE = "cpu"
AUDIO_PROMPT = "vim.wav"

model = ChatterboxTTS.from_pretrained(device=DEVICE)

def generate_audio(model, chunk, audio_prompt, cfg_weight, exaggeration):
    """Generate audio for a single chunk."""
    return model.generate(
        chunk,
        audio_prompt_path=audio_prompt,
        cfg_weight=cfg_weight,
        exaggeration=exaggeration,
    )

def save_audio(wav, output_path, sample_rate):
    """Save audio file to disk."""
    ta.save(output_path, wav, sample_rate)
    print(f"Saved: {output_path}")


@app.post("/generate")
def generate(text: str, file_id: str):
    # Create the full path
    file_path = RAM_DIR / f"{file_id}.wav"
    
    wav = generate_audio(
        model, text, "vim.wav", 0.1, 0.1
    )

    save_audio(wav, file_path, model.sr)
    
    return {"path": str(file_path)}

