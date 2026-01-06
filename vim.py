import os
import argparse
import json
from pathlib import Path
from chatterbox.tts_extended import ChatterboxTTS, parse_pause_tags, punc_norm
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
        "--device", type=str, required=True, help="Device to use (cpu or cuda)"
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


def process_manifest(args):
    """Process manifest file and regenerate invalid chunks."""
    manifest_file = Path(args.manifest_file)
    if not manifest_file.exists():
        print(f"Error: Manifest file '{args.manifest_file}' not found")
        return

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print(f"Loaded manifest from: {args.manifest_file}")

    output_dir = str(manifest_file.parent)

    invalid_entries = [entry for entry in manifest if not entry.get("valid", False)]

    if not invalid_entries:
        print("No invalid chunks found. All chunks are valid!")
        return

    print(f"Found {len(invalid_entries)} invalid chunk(s) to regenerate")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")

    model = ChatterboxTTS.from_pretrained(device=args.device)

    for entry in invalid_entries:
        chunk = entry["chunk"]
        audio_file = entry["audio_file"]

        print(f"Regenerating: {audio_file}")

        wav = generate_audio(
            model, chunk, args.audio_prompt, args.cfg_weight, args.exaggeration
        )

        output_path = os.path.join(output_dir, audio_file)
        save_audio(wav, output_path, model.sr)

        entry["retry"] = entry.get("retry", 0) + 1

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)

    print(f"\nRegenerated {len(invalid_entries)} chunk(s)")
    print(f"Updated manifest: {manifest_file}")


def process_text_file(args):
    """Process text file and generate audio chunks."""
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    text_file = Path(args.text_file)
    if not text_file.exists():
        print(f"Error: Text file '{args.text_file}' not found")
        return

    text = text_file.read_text().strip()
    print(f"Read text from: {args.text_file}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")

    model = ChatterboxTTS.from_pretrained(device=args.device)

    chunks = chunk_text(text)
    manifest = []

    for i, chunk in enumerate(chunks):
        wav = generate_audio(
            model, chunk, args.audio_prompt, args.cfg_weight, args.exaggeration
        )

        output_filename = f"{i+1:03d}.wav"
        output_path = os.path.join(output_dir, output_filename)
        save_audio(wav, output_path, model.sr)

        normalized_segments = []
        segments = parse_pause_tags(chunk)
        for segment, _ in segments:
            if segment.strip():
                normalized_segments.append(punc_norm(segment))

        manifest.append(
            {
                "chunk": chunk,
                "normalized_chunk": " ".join(normalized_segments),
                "audio_file": output_filename,
                "retry": 0,
                "valid": False,
            }
        )

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)

    print(f"\nAll files saved to '{output_dir}/' directory")
    print(f"Manifest saved to: {manifest_path}")


def main():
    args = parse_args()

    if args.manifest_file:
        process_manifest(args)
    else:
        process_text_file(args)


if __name__ == "__main__":
    main()
