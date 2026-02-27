"""The entry point for the fallout-codes package."""

from jax.lax import le

import argparse
import sys
import jax

from jax import random
import jax.numpy as jnp


def non_alphabetic_characters(length: int, key: jax.Array) -> list[str]:
    """Returns a list of non-alphabetic character codes of the specified length."""
    ordinals = random.choice(
        key,
        jnp.array([ord(c) for c in "!@#$%^&*()_+-=[]{}|;':\",.<>/?`~"]),
        shape=(length,),
        replace=True,
    ).tolist()
    return [chr(c) for c in ordinals]


def build_grid_text(length: int, words: list[str], key: jax.Array) -> list[str]:
    key_text, key_words, key_place = random.split(key, 3)
    text = non_alphabetic_characters(length, key_text)

    # This algorithm works by defining `word_count + 1` blocks of padding
    # (before, between, and after words). The total space not used by words is
    # randomly distributed among these padding blocks, ensuring at least one
    # character of padding between words.

    # 1. Calculate how much space is available for padding.
    total_word_len = sum(len(word) for word in words)
    total_padding_space = length - total_word_len
    min_internal_padding = max(0, len(words) - 1)
    extra_padding = total_padding_space - min_internal_padding

    if extra_padding < 0:
        raise ValueError(
            f"Text of length {length} is too short to contain {len(words)} "
            "words with the required spacing."
        )

    # 2. Randomly distribute extra padding among the (word_count + 1) blocks.
    key, subkey = random.split(key_place)
    partitions = jnp.sort(
        random.randint(subkey, shape=(len(words),), minval=0, maxval=extra_padding + 1)
    )
    all_partitions = jnp.concatenate(
        [jnp.array([0]), partitions, jnp.array([extra_padding])]
    )
    extra_pads = jnp.diff(all_partitions)

    # 3. Determine the final size of each padding block.
    base_pads = jnp.zeros(len(words) + 1, dtype=jnp.int32)
    if len(words) > 1:
        # Add the minimum 1-char space for internal padding blocks.
        base_pads = base_pads.at[1:-1].set(1)
    padding_sizes = base_pads + extra_pads

    # 4. Calculate word start indices from the padding sizes.
    cumulative_padding_before_word = jnp.cumsum(padding_sizes)[:-1]
    word_offsets = jnp.array([len(word) for word in words])
    start_indices = (
        cumulative_padding_before_word + jnp.cumsum(word_offsets) - word_offsets
    )

    # 5. Place the chosen words at the calculated start indices.
    for word, start in zip(words, start_indices.tolist()):
        text[start : start + len(word)] = list(word)

    return text


def build_grid_lines(
    width: int,
    height: int,
    words: list[str],
    key: jax.Array,
) -> list[str]:
    """Builds a grid of words and non-alphabetic characters."""
    key_text, key_place = random.split(key, 2)
    text = build_grid_text(width * height, words=words, key=key_text)

    # Split the text into lines of the specified width.
    lines = ["".join(text[i : i + width]) for i in range(0, len(text), width)]
    return lines


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
            "ABOUT",
            "ABOVE",
            "ABUSE",
            "ACTOR",
            "ACUTE",
            "ADMIT",
            "ADOPT",
            "ADULT",
            "AFTER",
            "AGAIN",
            "AGENT",
            "AGREE",
            "AHEAD",
            "ALARM",
            "ALBUM",
            "ALERT",
            "ALIKE",
            "ALIVE",
            "ALLOW",
            "ALONE",
            "ALONG",
            "ALTER",
            "AMONG",
            "ANGER",
            "ANGLE",
            "ANGRY",
            "APART",
            "APPLE",
            "APPLY",
            "ARENA",
            "ARGUE",
            "ARISE",
            "ARRAY",
            "ASIDE",
            "ASSET",
            "AUDIO",
            "AUDIT",
            "AVOID",
            "AWARD",
            "AWARE",
            "BADLY",
            "BAKER",
            "BASES",
            "BASIC",
            "BASIS",
            "BEACH",
            "BEGAN",
            "BEGIN",
            "BEGUN",
            "BEING",
            "BELOW",
            "BENCH",
            "BILLY",
            "BIRTH",
            "BLACK",
            "BLAME",
            "BLIND",
            "BLOCK",
            "BLOOD",
            "BOARD",
            "BRAIN",
            "BREAD",
            "BRUSH",
            "CHAIR",
            "CHARM",
            "CHEST",
            "CHORD",
            "CLICK",
            "CLOCK",
            "CLOUD",
            "CODES",
            "DANCE",
            "DEBUG",
            "DIARY",
            "DRINK",
            "EARTH",
            "FLUTE",
            "FRUIT",
            "GHOST",
            "GRAPE",
            "GREEN",
            "HAPPY",
            "HEART",
            "HOUSE",
            "INDEX",
            "INPUT",
            "JAXON",
            "JUICE",
            "LIGHT",
            "LOGIC",
            "MACRO",
            "MONEY",
            "MUSIC",
            "OTHER",
            "PARTY",
            "PIXEL",
            "PIZZA",
            "PLANT",
            "PROXY",
            "QUAKE",
            "QUERY",
            "RADIO",
            "RIVER",
            "SALAD",
            "SHEEP",
            "SHOES",
            "SMILE",
            "SNACK",
            "SNAKE",
            "SOLVE",
            "SPICE",
            "SPOON",
        ]

    if not all_words:
        print("Error: Word list is empty.", file=sys.stderr)
        sys.exit(1)

    # Determine word length from the first word and filter the list for consistency.
    word_length = len(all_words[0])
    # Also remove duplicates.
    all_words = sorted(list(set([w for w in all_words if len(w) == word_length])))

    if len(all_words) < args.word_count:
        print(
            f"Error: Not enough unique words of length {word_length} available. "
            f"Found {len(all_words)}, but need {args.word_count}.",
            file=sys.stderr,
        )
        sys.exit(1)

    key = random.PRNGKey(args.seed)

    # try:
    key_words, *grid_keys = random.split(key, args.grid_count + 1)
    # 0. Randomly select `word_count` words from the list without replacement.
    chosen_word_indices = random.choice(
        key_words,
        jnp.arange(len(all_words)),
        shape=(args.grid_count * args.word_count,),
        replace=False,
    )
    chosen_words = [all_words[i] for i in chosen_word_indices]

    grids: list[list[str]] = []
    for i, grid_key in enumerate(grid_keys):
        grid = build_grid_lines(
            width=args.width,
            height=args.height,
            words=chosen_words[i::args.grid_count],
            key=grid_key,
        )
        grids.append(grid)
    screen_lines = []
    for grid_lines in zip(*grids):
        screen_lines.append(" | ".join(grid_lines))
    print("\n".join(f"| {line} |" for line in screen_lines))
    # except ValueError as e:
    #     print(f"Error: {e}", file=sys.stderr)
    #     sys.exit(1)


if __name__ == "__main__":
    main()
