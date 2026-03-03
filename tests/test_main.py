"""Tests for the main module."""

import argparse
from xyz.fallout_codes.__main__ import setup_game
from xyz.fallout_codes.game import Game


def test_setup_game() -> None:
    """Test the setup_game function."""
    # Create a dummy args object
    args = argparse.Namespace(
        seed=42,
        word_file=None,
        word_count=8,
        word_length=5,
        grid_count=2,
        width=12,
        height=16,
    )

    game, screen_lines = setup_game(args)

    assert isinstance(game, Game)
    assert len(game.candidate_words) == args.word_count * args.grid_count
    assert len(game.target_password) == args.word_length
    assert game.target_password in game.candidate_words

    assert isinstance(screen_lines, list)
    assert len(screen_lines) == args.height

    # Check the width of the screen lines
    expected_line_width = args.width * args.grid_count + (args.grid_count - 1) * len(
        " | "
    )
    for line in screen_lines:
        assert len(line) == expected_line_width

    # Separate the two grids
    grid1_lines = [line.split(" | ")[0] for line in screen_lines]
    grid2_lines = [line.split(" | ")[1] for line in screen_lines]
    grid1_text = "".join(grid1_lines)
    grid2_text = "".join(grid2_lines)

    # Check that the words are in the grids
    words_grid1 = game.candidate_words[0::2]
    words_grid2 = game.candidate_words[1::2]

    for word in words_grid1:
        assert word in grid1_text
    for word in words_grid2:
        assert word in grid2_text
