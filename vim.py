from chatterbox.tts_extended import ChatterboxTTS
import torchaudio as ta
from utils.text import chunk_text

device = "cpu"
AUDIO_PROMPT_PATH = "vim.wav"
CFG_WEIGHT = 0.1
EXAGGERATION = 0.1

model = ChatterboxTTS.from_pretrained(device=device)

text = "hello[pause:4.0s] world"

chunks = chunk_text(text)

for i, chunk in enumerate(chunks):
    wav = model.generate(
        chunk,
        audio_prompt_path=AUDIO_PROMPT_PATH,
        cfg_weight=CFG_WEIGHT,
        exaggeration=EXAGGERATION,
    )
    ta.save(f"test-{i}.wav", wav, model.sr)
