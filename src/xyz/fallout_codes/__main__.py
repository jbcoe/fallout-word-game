"""The entry point for the fallout-codes package."""

import argparse
import sys
from jax import random
import jax.numpy as jnp

from xyz.fallout_codes.words import _SAMPLE_WORDS
from xyz.fallout_codes.grid import build_grid_lines


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a Fallout-style code-breaking grid."
    )
    parser.add_argument(
        "--seed", type=int, default=65, help="Seed for the random number generator."
    )
    parser.add_argument(
        "--word-file",
        type=str,
        default=None,
        help="Path to a file containing a list of words, one per line. If not provided, a default list is used.",
    )
    parser.add_argument(
        "--word-count",
        type=int,
        default=8,
        help="Number of words to place on the grid.",
    )
    parser.add_argument(
        "--word-length",
        type=int,
        default=5,
        help="Length of each word to place on the grid.",
    )
    parser.add_argument(
        "--grid-count",
        type=int,
        default=2,
        help="Number of grids to generate (default is 2).",
    )
    parser.add_argument(
        "--width", type=int, default=12, help="Width of the character grid."
    )
    parser.add_argument(
        "--height", type=int, default=16, help="Height of the character grid."
    )
    return parser.parse_args()


def load_words(word_file: str | None) -> list[str]:
    """Loads a list of words from a file, or returns the default list."""
    if word_file:
        with open(word_file, "r") as f:
            words = [line.strip().upper() for line in f if line.strip()]
        return words

    return _SAMPLE_WORDS


def filter_words(all_words: list[str], word_length: int) -> list[str]:
    """Filters words by length and removes duplicates."""
    return sorted(list(set([w for w in all_words if len(w) == word_length])))


def main() -> None:
    """Runs the grid generator."""
    args = parse_args()

    if args.word_length <= 0:
        raise ValueError("Error: Word length must be a positive integer.")

    if args.grid_count <= 0:
        raise ValueError("Error: Grid count must be a positive integer.")

    if args.word_count <= 0:
        raise ValueError("Error: Word count must be a positive integer.")

    all_words = load_words(args.word_file)

    filtered_words = filter_words(all_words, args.word_length)

    required_word_count = args.word_count * args.grid_count
    if len(filtered_words) < required_word_count:
        print(
            f"Error: Not enough unique words of length {args.word_length} available. "
            f"Found {len(filtered_words)}, but need {required_word_count}.",
            file=sys.stderr,
        )
        sys.exit(1)

    key = random.PRNGKey(args.seed)

    words_key, *grid_keys = random.split(key, args.grid_count + 1)

    # Randomly select words without replacement
    chosen_word_indices = random.choice(
        words_key,
        jnp.arange(len(filtered_words)),
        shape=(required_word_count,),
        replace=False,
    )
    chosen_words = [filtered_words[i] for i in chosen_word_indices]

    grids: list[list[str]] = []
    for i, grid_key in enumerate(grid_keys):
        # Slice the chosen words for this specific grid
        grid_words = chosen_words[i :: args.grid_count]

        grid = build_grid_lines(
            width=args.width,
            height=args.height,
            words=grid_words,
            key=grid_key,
        )
        grids.append(grid)

    screen_lines = []
    for grid_lines in zip(*grids):
        screen_lines.append(" | ".join(grid_lines))
    print("\n".join(f"| {line} |" for line in screen_lines))


if __name__ == "__main__":
    main()
