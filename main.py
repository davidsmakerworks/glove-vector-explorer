"""GloVe Vector Explorer — interactive CLI for walking through embedding space."""

import argparse
import sys

import numpy as np

from vectors import load_vectors, unit_matrix, nearest, parse_line

USAGE_HINT = (
    "  Usage: [+|-] word   add or subtract a word vector\n"
    "         word          jump to a new starting word\n"
    "         /exit         quit"
)


def prompt_vector_file() -> str:
    path = input("GloVe file path (.txt or .npz): ").strip()
    if not path:
        print("No file provided. Exiting.")
        sys.exit(1)
    return path


def prompt_start_word(index: dict[str, int]) -> str:
    while True:
        word = input("Starting word: ").strip().lower()
        if not word:
            continue
        if word in index:
            return word
        print(f"  '{word}' not in vocabulary. Try another word.")


def repl(
    words: list[str],
    mat: np.ndarray,
    unit_mat: np.ndarray,
    index: dict[str, int],
    start: str,
) -> None:
    current = mat[index[start]].copy()
    used: set[int] = {index[start]}
    expression = start
    print(f"  Starting at '{start}'. Enter '+ word', '- word', or a bare word to jump.")

    while True:
        try:
            line = input(f"\n[{expression}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return

        if not line:
            continue

        if line.lower() == "/exit":
            print("Bye!")
            return

        parsed = parse_line(line)
        if parsed is None:
            print(USAGE_HINT)
            continue

        op, word = parsed

        if word not in index:
            print(f"  '{word}' not in vocabulary.")
            continue

        if op == "=":
            current = mat[index[word]].copy()
            used = {index[word]}
            expression = word
            print(f"  Jumped to '{word}'.")
            continue

        vec = mat[index[word]]
        if op == "+":
            current = current + vec
            expression = f"{expression} + {word}"
        else:
            current = current - vec
            expression = f"{expression} - {word}"

        used.add(index[word])

        result_word, sim = nearest(current, unit_mat, words, used)[0]
        print(f"  → {result_word}  (similarity {sim:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk through GloVe embedding space interactively.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python main.py --vectors glove.6B.100d.txt --cache glove.npz\n"
            "  python main.py --vectors glove.npz"
        ),
    )
    parser.add_argument(
        "--vectors", "-v",
        metavar="FILE",
        help="GloVe text file (.txt) or cached vectors file (.npz)",
    )
    parser.add_argument(
        "--cache", "-c",
        metavar="FILE",
        help="Save loaded vectors to this .npz file for faster future loads",
    )
    args = parser.parse_args()

    path = args.vectors or prompt_vector_file()

    print(f"Loading vectors from '{path}'...")
    try:
        words, mat, index = load_vectors(path, cache_path=args.cache)
    except FileNotFoundError:
        print(f"File not found: {path}")
        sys.exit(1)
    except Exception as e:
        print(f"Failed to load vectors: {e}")
        sys.exit(1)

    if args.cache and not path.endswith(".npz"):
        print(f"Cache saved to '{args.cache}'.")

    umat = unit_matrix(mat)
    print(f"Loaded {len(words):,} words, {mat.shape[1]}-dimensional vectors.")

    start = prompt_start_word(index)
    repl(words, mat, umat, index, start)


if __name__ == "__main__":
    main()
