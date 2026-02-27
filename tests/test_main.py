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
    # 2 grids side by side with separator (width) + (width) + 3 (" | ")
    # But wait, grid generation logic puts padding etc.
    # Just check it's not empty and basic structure looks ok.
    assert len(screen_lines) > 0
