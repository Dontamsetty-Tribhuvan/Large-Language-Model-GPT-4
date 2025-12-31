import sentencepiece as spm
import argparse

def train(input_file, model_prefix, vocab_size=50280):
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        user_defined_symbols=[
            "<|system|>", "<|user|>", "<|assistant|>",
            "<|tool|>", "<|think|>", "<|reflect|>", "<|final|>"
        ]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="tokenizer")
    args = parser.parse_args()
    train(args.input, args.out)
