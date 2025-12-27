import os
import argparse
from pathlib import Path
from chatterbox.tts_extended import ChatterboxTTS, parse_pause_tags
import torchaudio as ta
from utils.text import chunk_text


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate audio from text file or manifest"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text-file", type=str, help="Path to text file")
    group.add_argument("--manifest-file", type=str, help="Path to manifest file")

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="out",
        help="Output directory for wav files (default: out)",
    )
    parser.add_argument(
        "--device", type=str, required=True, help="Device to use (default: cpu)"
    )
    parser.add_argument(
        "--audio-prompt",
        type=str,
        default="vim.wav",
        help="Path to audio prompt file (default: vim.wav)",
    )
    parser.add_argument(
        "--cfg-weight", type=float, default=0.1, help="CFG weight (default: 0.1)"
    )
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.1,
        help="Exaggeration parameter (default: 0.1)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Handle manifest file
    if args.manifest_file:
        manifest_file = Path(args.manifest_file)
        if not manifest_file.exists():
            print(f"Error: Manifest file '{args.manifest_file}' not found")
            return

        manifest_content = manifest_file.read_text()
        print(f"Manifest content from: {args.manifest_file}")
        print(manifest_content)
        return

    # Handle text file (existing logic)
    OUTPUT_DIR = args.output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read text from file
    text_file = Path(args.text_file)
    if not text_file.exists():
        print(f"Error: Text file '{args.text_file}' not found")
        return

    text = text_file.read_text().strip()
    print(f"Read text from: {args.text_file}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Device: {args.device}")

    model = ChatterboxTTS.from_pretrained(device=args.device)

    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        segments = parse_pause_tags(chunk)
        wav = model.generate(
            segments,
            audio_prompt_path=args.audio_prompt,
            cfg_weight=args.cfg_weight,
            exaggeration=args.exaggeration,
        )

        output_path = os.path.join(OUTPUT_DIR, f"{i+1:03d}.wav")
        ta.save(output_path, wav, model.sr)
        print(f"Saved: {output_path}")

    print(f"\nAll files saved to '{OUTPUT_DIR}/' directory")


if __name__ == "__main__":
    main()
