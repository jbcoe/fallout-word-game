"""The entry point for the fallout-codes package."""

import argparse
import sys

from jax import random
import jax.numpy as jnp


def non_alphabetic_characters(length: int, key: random.PRNGKey) -> list[str]:
    """Returns a list of non-alphabetic character codes of the specified length."""
    ordinals = random.choice(
        key,
        jnp.array([ord(c) for c in "!@#$%^&*()_+-=[]{}|;':\",.<>/?`~"]),
        shape=(length,),
        replace=True,
    ).tolist()
    return [chr(c) for c in ordinals]


def split_into_lines(text: list[str], width: int) -> list[str]:
    """Splits the text into lines of the specified width."""
    return ["".join(text[i : i + width]) for i in range(0, len(text), width)]


def run(
    width: int,
    height: int,
    word_count: int,
    word_length: int,
    all_words: list[str],
    key: random.PRNGKey,
) -> None:
    """The main logic for generating and printing the grid."""
    key_a, key_b, key_words, key_place = random.split(key, 4)

    text_a = non_alphabetic_characters(width * height, key_a)
    text_b = non_alphabetic_characters(width * height, key_b)

    # 0. Randomly select `word_count` words from the list without replacement.
    chosen_word_indices = random.choice(
        key_words, jnp.arange(len(all_words)), shape=(word_count,), replace=False
    )
    chosen_words = [all_words[i] for i in chosen_word_indices]

    # This algorithm works by defining `word_count + 1` blocks of padding
    # (before, between, and after words). The total space not used by words is
    # randomly distributed among these padding blocks, ensuring at least one
    # character of padding between words.

    # 1. Calculate how much space is available for padding.
    total_word_len = word_count * word_length
    total_padding_space = (width * height) - total_word_len
    min_internal_padding = max(0, word_count - 1)
    extra_padding = total_padding_space - min_internal_padding

    if extra_padding < 0:
        raise ValueError(
            f"Text of length {width * height} is too short to contain {word_count} "
            f"words of length {word_length} with required spacing."
        )

    # 2. Randomly distribute extra padding among the (word_count + 1) blocks.
    key, subkey = random.split(key_place)
    partitions = jnp.sort(
        random.randint(
            subkey, shape=(word_count,), minval=0, maxval=extra_padding + 1
        )
    )
    all_partitions = jnp.concatenate(
        [jnp.array([0]), partitions, jnp.array([extra_padding])]
    )
    extra_pads = jnp.diff(all_partitions)

    # 3. Determine the final size of each padding block.
    base_pads = jnp.zeros(word_count + 1, dtype=jnp.int32)
    if word_count > 1:
        # Add the minimum 1-char space for internal padding blocks.
        base_pads = base_pads.at[1:-1].set(1)
    padding_sizes = base_pads + extra_pads

    # 4. Calculate word start indices from the padding sizes.
    cumulative_padding_before_word = jnp.cumsum(padding_sizes)[:-1]
    word_offsets = jnp.arange(word_count) * word_length
    start_indices = cumulative_padding_before_word + word_offsets

    # 5. Place the chosen words at the calculated start indices.
    for word, start in zip(chosen_words, start_indices.tolist()):
        text_a[start : start + word_length] = list(word)

    # Split the text into lines of the specified width.
    lines_a = split_into_lines(text_a, width)
    lines_b = split_into_lines(text_b, width)

    # Print the lines.
    print("\n".join(f"| {a} | {b} |" for a, b in zip(lines_a, lines_b)))


def main() -> None:
    """Parses command-line arguments and runs the grid generator."""
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
        "--width", type=int, default=12, help="Width of the character grid."
    )
    parser.add_argument(
        "--height", type=int, default=16, help="Height of the character grid."
    )
    args = parser.parse_args()

    if args.word_file:
        try:
            with open(args.word_file, "r") as f:
                all_words = [line.strip().upper() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Error: Word file not found at {args.word_file}", file=sys.stderr)
            sys.exit(1)
    else:
        all_words = [
            "ABOUT", "ABOVE", "ABUSE", "ACTOR", "ACUTE", "ADMIT", "ADOPT",
            "ADULT", "AFTER", "AGAIN", "AGENT", "AGREE", "AHEAD", "ALARM",
            "ALBUM", "ALERT", "ALIKE", "ALIVE", "ALLOW", "ALONE", "ALONG",
            "ALTER", "AMONG", "ANGER", "ANGLE", "ANGRY", "APART", "APPLE",
            "APPLY", "ARENA", "ARGUE", "ARISE", "ARRAY", "ASIDE", "ASSET",
            "AUDIO", "AUDIT", "AVOID", "AWARD", "AWARE", "BADLY", "BAKER",
            "BASES", "BASIC", "BASIS", "BEACH", "BEGAN", "BEGIN", "BEGUN",
            "BEING", "BELOW", "BENCH", "BILLY", "BIRTH", "BLACK", "BLAME",
            "BLIND", "BLOCK", "BLOOD", "BOARD", "BRAIN", "BREAD", "BRUSH",
            "CHAIR", "CHARM", "CHEST", "CHORD", "CLICK", "CLOCK", "CLOUD",
            "CODES", "DANCE", "DEBUG", "DIARY", "DRINK", "EARTH", "FLUTE",
            "FRUIT", "GHOST", "GRAPE", "GREEN", "HAPPY", "HEART", "HOUSE",
            "INDEX", "INPUT", "JAXON", "JUICE", "LIGHT", "LOGIC", "MACRO",
            "MONEY", "MUSIC", "OTHER", "PARTY", "PIXEL", "PIZZA", "PLANT",
            "PROXY", "QUAKE", "QUERY", "RADIO", "RIVER", "SALAD", "SHEEP",
            "SHOES", "SMILE", "SNACK", "SNAKE", "SOLVE", "SPICE", "SPOON",
        ]

    if not all_words:
        print("Error: Word list is empty.", file=sys.stderr)
        sys.exit(1)

    # Determine word length from the first word and filter the list for consistency.
    word_length = len(all_words[0])
    # Also remove duplicates.
    all_words = sorted(
        list(set([w for w in all_words if len(w) == word_length]))
    )

    if len(all_words) < args.word_count:
        print(
            f"Error: Not enough unique words of length {word_length} available. "
            f"Found {len(all_words)}, but need {args.word_count}.",
            file=sys.stderr,
        )
        sys.exit(1)

    key = random.PRNGKey(args.seed)

    try:
        run(
            width=args.width,
            height=args.height,
            word_count=args.word_count,
            word_length=word_length,
            all_words=all_words,
            key=key,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
