import jax
from jax import random
import jax.numpy as jnp

_NON_ALPHABETIC_CHARACTERS = jnp.array(
    [ord(c) for c in "!@#$%^&*()_+-=[]{}|;':\",.<>/?`~"]
)


def non_alphabetic_characters(length: int, key: jax.Array) -> list[str]:
    """Returns a list of non-alphabetic character codes of the specified length."""
    ordinals = random.choice(
        key,
        a=_NON_ALPHABETIC_CHARACTERS,
        shape=(length,),
        replace=True,
    ).tolist()
    return [chr(c) for c in ordinals]


def build_grid_text(length: int, words: list[str], key: jax.Array) -> list[str]:
    """Generates a text buffer containing hidden words and random characters."""
    text_key, key_words, place_key = random.split(key, 3)
    text = non_alphabetic_characters(length, text_key)

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
    key, subkey = random.split(place_key)
    # We need len(words) + 1 partitions to create slots around the words
    num_blocks = len(words) + 1

    # Generate random cut points to partition the extra padding
    partitions = jnp.sort(
        random.randint(
            subkey, shape=(num_blocks - 1,), minval=0, maxval=extra_padding + 1
        )
    )
    all_partitions = jnp.concatenate(
        [jnp.array([0]), partitions, jnp.array([extra_padding])]
    )
    extra_pads = jnp.diff(all_partitions)

    # 3. Determine the final size of each padding block.
    # Start with base padding: 0 for outer blocks, 1 for inner blocks
    base_pads = jnp.zeros(num_blocks, dtype=jnp.int32)
    if len(words) > 1:
        # Add the minimum 1-char space for internal padding blocks.
        base_pads = base_pads.at[1:-1].set(1)

    padding_sizes = base_pads + extra_pads

    # 4. Calculate word start indices.
    # The start index of a word is equal to the cumulative sum of the padding blocks before it
    # plus the cumulative sum of the lengths of the words before it.

    cumulative_padding_before_word = jnp.cumsum(padding_sizes)[:-1]
    word_offsets = jnp.pad(
        jnp.cumsum(jnp.array([len(w) for w in words]))[:-1], (1, 0), constant_values=0
    )

    start_indices = cumulative_padding_before_word + word_offsets

    # 5. Place the words.
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
    text = build_grid_text(width * height, words=words, key=key)

    # Split the text into lines of the specified width.
    lines = ["".join(text[i : i + width]) for i in range(0, len(text), width)]
    return lines
