import re
from typing import List
from chatterbox.tts_extended import ChatterboxTTS
import torchaudio as ta

device = "cpu"
AUDIO_PROMPT_PATH = "vim.wav"
CFG_WEIGHT = 0.1
EXAGGERATION = 0.1

model = ChatterboxTTS.from_pretrained(device=device)


# TODO: take this to utils package
def chunk_text(text: str, max_words: int = 60) -> List[str]:
    """
    Chunk text into segments based on sentence boundaries, respecting max word count.

    Sentences end with either:
    - Period followed by space/newline: '. '
    - Period followed by double quote: '."'

    Args:
        text: Input text to chunk
        max_words: Maximum words per chunk

    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []

    # Pattern to split on sentence boundaries
    # Matches: '. ' or '." ' or end of string after '. ' or '."'
    sentence_pattern = r"(?<=\.)\s+|(?<=\.\")\s+"

    # Split into sentences
    sentences = re.split(sentence_pattern, text.strip())

    # Clean up sentences (remove empty strings and strip whitespace)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())

        # If single sentence exceeds max_words, add it as its own chunk
        if sentence_word_count > max_words:
            # First, flush current chunk if it has content
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0
            # Add the long sentence as its own chunk
            chunks.append(sentence)
            continue

        # Check if adding this sentence would exceed max_words
        if current_word_count + sentence_word_count > max_words:
            # Flush current chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            # Start new chunk with current sentence
            current_chunk = [sentence]
            current_word_count = sentence_word_count
        else:
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_word_count += sentence_word_count

    # Add remaining chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


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
