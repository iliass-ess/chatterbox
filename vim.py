import os
from chatterbox.tts_extended import ChatterboxTTS, parse_pause_tags
import torchaudio as ta
from utils.text import chunk_text

device = "cpu"
AUDIO_PROMPT_PATH = "vim.wav"
CFG_WEIGHT = 0.1
EXAGGERATION = 0.1
OUTPUT_DIR = "out"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ChatterboxTTS.from_pretrained(device=device)

text = "hello[pause:4.0s] world"

chunks = chunk_text(text)

for i, chunk in enumerate(chunks):
    segments = parse_pause_tags(chunk)
    wav = model.generate(
        segments,
        audio_prompt_path=AUDIO_PROMPT_PATH,
        cfg_weight=CFG_WEIGHT,
        exaggeration=EXAGGERATION,
    )
    
    output_path = os.path.join(OUTPUT_DIR, f"test-{i}.wav")
    ta.save(output_path, wav, model.sr)
    print(f"Saved: {output_path}")

print(f"\nAll files saved to '{OUTPUT_DIR}/' directory")
