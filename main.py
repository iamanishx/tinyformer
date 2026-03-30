import torch

from config import DECODER_ONLY_CONFIG, ENCODER_DECODER_CONFIG
from models import DecoderOnlyTransformer, EncoderDecoderTransformer


def demo_encoder_decoder() -> None:
    print("=" * 60)
    print("1. Encoder-Decoder Transformer")
    print("=" * 60)

    model = EncoderDecoderTransformer(**ENCODER_DECODER_CONFIG)

    batch_size = 2
    src = torch.randint(0, ENCODER_DECODER_CONFIG["src_vocab_size"], (batch_size, 10))
    tgt = torch.randint(0, ENCODER_DECODER_CONFIG["tgt_vocab_size"], (batch_size, 7))

    logits = model(src, tgt)
    print(f"src shape:    {src.shape}")
    print(f"tgt shape:    {tgt.shape}")
    print(f"logits shape: {logits.shape}")


def demo_decoder_only() -> None:
    print("=" * 60)
    print("2. Decoder-Only Transformer")
    print("=" * 60)

    model = DecoderOnlyTransformer(**DECODER_ONLY_CONFIG)

    batch_size = 2
    tokens = torch.randint(0, DECODER_ONLY_CONFIG["vocab_size"], (batch_size, 10))
    logits = model(tokens)

    print(f"input shape:  {tokens.shape}")
    print(f"logits shape: {logits.shape}")

    prompt = torch.randint(0, DECODER_ONLY_CONFIG["vocab_size"], (1, 5))
    generated = model.generate(prompt, max_new_tokens=5)
    print(f"generated shape: {generated.shape}")


def main() -> None:
    torch.manual_seed(42)
    demo_encoder_decoder()
    print()
    demo_decoder_only()


if __name__ == "__main__":
    main()
