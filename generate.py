"""Generate text from a trained character-level decoder-only transformer.

Example:
    python3 generate.py --prompt "Once upon a time"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models import DecoderOnlyTransformer


CHECKPOINT_PATH = Path("checkpoints/decoder_only_char.pt")
META_PATH = Path("checkpoints/decoder_only_char_meta.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="hello")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser.parse_args()


def load_metadata() -> dict:
    if not META_PATH.exists() or not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "Missing checkpoint files. Run train.py first to create them."
        )
    return json.loads(META_PATH.read_text(encoding="utf-8"))


def encode_prompt(prompt: str, stoi: dict[str, int]) -> torch.Tensor:
    unknown = [ch for ch in prompt if ch not in stoi]
    if unknown:
        missing = "".join(sorted(set(unknown)))
        raise ValueError(
            f"Prompt contains characters not seen during training: {missing!r}"
        )
    return torch.tensor([[stoi[ch] for ch in prompt]], dtype=torch.long)


def decode_tokens(tokens: torch.Tensor, itos: dict[int, str]) -> str:
    return "".join(itos[int(idx)] for idx in tokens.tolist())


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = load_metadata()
    stoi = {ch: idx for ch, idx in meta["stoi"].items()}
    itos = {int(idx): ch for idx, ch in meta["itos"].items()}
    model_config = meta["model_config"]

    model = DecoderOnlyTransformer(**model_config).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    prompt = encode_prompt(args.prompt, stoi).to(device)
    generated = model.generate(
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    text = decode_tokens(generated[0].cpu(), itos)
    print(text)


if __name__ == "__main__":
    main()
