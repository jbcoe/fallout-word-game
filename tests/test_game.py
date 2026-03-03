"""Tests for the game module."""

import pytest

from xyz.fallout_codes.game import Game, calculate_likeness


def test_calculate_likeness() -> None:
    """Test the calculate_likeness function."""
    assert calculate_likeness("ABCD", "ABCD") == 4
    assert calculate_likeness("ABCD", "ABCE") == 3
    assert calculate_likeness("ABCD", "EFGH") == 0
    assert calculate_likeness("SCORPION", "VAMPIRE") == 0
    assert calculate_likeness("SCORPION", "SCORPION") == 8
    assert calculate_likeness("SARGIOAA", "SCORPION") == 1


def test_game_initialization() -> None:
    """Test game initialization."""
    candidates = ["WORD1", "WORD2", "WORD3"]
    game = Game(target_password="WORD1", candidate_words=candidates)
    assert game.target_password == "WORD1"
    assert game.attempts_left == 4
    assert not game.is_game_over
    assert not game.has_won
    assert game.history == []


def test_game_init_raises_error_if_target_not_in_candidates() -> None:
    """Test that ValueError is raised if target is not in candidates."""
    with pytest.raises(
        ValueError, match="Target password must be one of the candidate words."
    ):
        Game(target_password="INVALID", candidate_words=["WORD1", "WORD2"])


def test_game_init_raises_error_if_attempts_is_not_positive() -> None:
    """Test that ValueError is raised if attempts_left is not positive."""
    with pytest.raises(ValueError, match="Attempts left must be a positive integer."):
        Game(
            target_password="WORD1", candidate_words=["WORD1", "WORD2"], attempts_left=0
        )


def test_game_correct_guess() -> None:
    """Test a correct guess."""
    candidates = ["WORD1", "WORD2"]
    game = Game(target_password="WORD1", candidate_words=candidates)
    result = game.make_guess("WORD1")
    assert result == "Correct."
    assert game.has_won
    assert game.is_game_over
    assert game.history == [("WORD1", 5)]


def test_game_incorrect_guess() -> None:
    """Test an incorrect guess."""
    candidates = ["WORD1", "WORD2"]
    game = Game(target_password="WORD1", candidate_words=candidates)
    result = game.make_guess("WORD2")
    assert "Likeness=4" in result
    assert not game.has_won
    assert not game.is_game_over
    assert game.attempts_left == 3
    assert game.history == [("WORD2", 4)]


def test_game_invalid_guess() -> None:
    """Test a guess not in candidate list."""
    game = Game(target_password="ABCD", candidate_words=["ABCD", "EFGH"])
    result = game.make_guess("XYZW")
    assert result == "Entry denied."
    assert not game.has_won
    assert not game.is_game_over
    assert game.attempts_left == 4
    assert game.history == []


def test_game_over_after_exhausting_attempts() -> None:
    """Test running out of attempts."""
    game = Game(
        target_password="ABCD", candidate_words=["ABCD", "EFGH", "IJKL", "MNOP", "QRST"]
    )
    game.make_guess("EFGH")
    game.make_guess("IJKL")
    game.make_guess("MNOP")
    result = game.make_guess("QRST")
    assert "Lockdown initiated" in result
    assert not game.has_won
    assert game.is_game_over
    assert game.attempts_left == 0


def test_game_over_when_starting_with_one_attempt() -> None:
    """Test losing the game when starting with only one attempt."""
    candidates = ["WORD1", "WORD2"]
    game = Game(target_password="WORD1", candidate_words=candidates, attempts_left=1)
    result = game.make_guess("WORD2")
    assert "Lockdown initiated" in result
    assert game.attempts_left == 0
    assert game.is_game_over
    assert not game.has_won


def test_make_guess_when_game_is_over() -> None:
    """Test making a guess when the game is already over."""
    game = Game(target_password="WORD1", candidate_words=["WORD1", "WORD2"])
    game.is_game_over = True
    initial_attempts = game.attempts_left
    initial_history = list(game.history)

    result = game.make_guess("WORD1")
    assert result == "Game over."
    assert game.attempts_left == initial_attempts
    assert game.history == initial_history
