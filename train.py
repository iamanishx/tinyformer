"""Train a tiny character-level decoder-only transformer.

Expected data file:
    data/input.txt

Example:
    python3 train.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from models import DecoderOnlyTransformer


DATA_PATH = Path("data/input.txt")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "decoder_only_char.pt"
META_PATH = CHECKPOINT_DIR / "decoder_only_char_meta.json"


def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing training data at {path}. Create data/input.txt first."
        )
    return path.read_text(encoding="utf-8")


def build_vocab(text: str) -> tuple[list[str], dict[str, int], dict[int, str]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return chars, stoi, itos


def encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def get_batch(
    data: torch.Tensor, batch_size: int, block_size: int, device: torch.device
):
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


def estimate_loss(
    model: DecoderOnlyTransformer,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: torch.device,
) -> dict[str, float]:
    out = {}
    model.eval()
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size, device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 32
    block_size = 128
    max_iters = 2000
    eval_interval = 200
    eval_iters = 50
    learning_rate = 3e-4

    text = load_text(DATA_PATH)
    chars, stoi, itos = build_vocab(text)
    data = encode(text, stoi)

    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    model = DecoderOnlyTransformer(
        vocab_size=len(chars),
        d_model=128,
        num_heads=4,
        num_layers=4,
        d_ff=512,
        max_len=block_size,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"device: {device}")
    print(f"text length: {len(text):,}")
    print(f"vocab size: {len(chars)}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    for step in range(max_iters):
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                batch_size,
                block_size,
                eval_iters,
                device,
            )
            print(
                f"step {step:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}"
            )

        x, y = get_batch(train_data, batch_size, block_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    META_PATH.write_text(
        json.dumps(
            {
                "stoi": stoi,
                "itos": {str(i): ch for i, ch in itos.items()},
                "block_size": block_size,
                "model_config": {
                    "vocab_size": len(chars),
                    "d_model": 128,
                    "num_heads": 4,
                    "num_layers": 4,
                    "d_ff": 512,
                    "max_len": block_size,
                    "dropout": 0.1,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved model to {CHECKPOINT_PATH}")
    print(f"saved metadata to {META_PATH}")


if __name__ == "__main__":
    main()
