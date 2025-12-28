import os
import argparse
import json
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

        # Read and parse manifest
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        print(f"Loaded manifest from: {args.manifest_file}")

        # Get output directory from manifest file location
        OUTPUT_DIR = str(manifest_file.parent)

        # Filter invalid chunks
        invalid_entries = [entry for entry in manifest if not entry.get("valid", False)]

        if not invalid_entries:
            print("No invalid chunks found. All chunks are valid!")
            return

        print(f"Found {len(invalid_entries)} invalid chunk(s) to regenerate")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Device: {args.device}")

        # Load model
        model = ChatterboxTTS.from_pretrained(device=args.device)

        # Regenerate invalid chunks
        for entry in invalid_entries:
            chunk = entry["chunk"]
            audio_file = entry["audio_file"]

            print(f"Regenerating: {audio_file}")

            segments = parse_pause_tags(chunk)
            wav = model.generate(
                segments,
                audio_prompt_path=args.audio_prompt,
                cfg_weight=args.cfg_weight,
                exaggeration=args.exaggeration,
            )

            output_path = os.path.join(OUTPUT_DIR, audio_file)
            ta.save(output_path, wav, model.sr)
            print(f"Saved: {output_path}")

            # Increment retry count
            entry["retry"] = entry.get("retry", 0) + 1

        # Save updated manifest
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)

        print(f"\nRegenerated {len(invalid_entries)} chunk(s)")
        print(f"Updated manifest: {manifest_file}")
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
    manifest = []

    for i, chunk in enumerate(chunks):
        segments = parse_pause_tags(chunk)
        wav = model.generate(
            segments,
            audio_prompt_path=args.audio_prompt,
            cfg_weight=args.cfg_weight,
            exaggeration=args.exaggeration,
        )

        output_filename = f"{i+1:03d}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        ta.save(output_path, wav, model.sr)
        print(f"Saved: {output_path}")

        # Add entry to manifest
        manifest.append(
            {"chunk": chunk, "audio_file": output_filename, "retry": 0, "valid": False}
        )

    # Save manifest.json
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)

    print(f"\nAll files saved to '{OUTPUT_DIR}/' directory")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
