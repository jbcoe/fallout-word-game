"""Tests for the game module."""

from xyz.fallout_codes.game import Game, calculate_likeness


def test_calculate_likeness() -> None:
    """Test the calculate_likeness function."""
    assert calculate_likeness("ABCD", "ABCD") == 4
    assert calculate_likeness("ABCD", "ABCE") == 3
    assert calculate_likeness("ABCD", "EFGH") == 0
    assert calculate_likeness("SCORPION", "VAMPIRE") == 0  # Different length, no match
    assert calculate_likeness("SCORPION", "SCORPION") == 8
    # "SARGIOAA" vs "SCORPION":
    # S vs S (Match)
    # A vs C
    # R vs O
    # G vs R
    # I vs P
    # O vs I
    # A vs O


def test_game_initialization() -> None:
    """Test game initialization."""
    game = Game(target_password="ABCD", candidate_words=["ABCD", "EFGH"])
    assert game.attempts_left == 4
    assert not game.is_game_over
    assert not game.has_won
    assert game.history == []


def test_game_correct_guess() -> None:
    """Test a correct guess."""
    game = Game(target_password="ABCD", candidate_words=["ABCD", "EFGH"])
    result = game.make_guess("ABCD")
    assert result == "Correct."
    assert game.has_won
    assert game.is_game_over
    assert game.history == [("ABCD", 4)]


def test_game_incorrect_guess() -> None:
    """Test an incorrect guess."""
    game = Game(target_password="ABCD", candidate_words=["ABCD", "EFGH"])
    result = game.make_guess("EFGH")
    assert "Likeness=0" in result
    assert not game.has_won
    assert not game.is_game_over
    assert game.attempts_left == 3
    assert game.history == [("EFGH", 0)]


def test_game_invalid_guess() -> None:
    """Test a guess not in candidate list."""
    game = Game(target_password="ABCD", candidate_words=["ABCD", "EFGH"])
    result = game.make_guess("XYZW")
    assert result == "Entry denied."
    assert not game.has_won
    assert not game.is_game_over
    # Invalid guess should not consume an attempt
    assert game.attempts_left == 4
    assert game.history == []


def test_game_over_attempts() -> None:
    """Test running out of attempts."""
    game = Game(
        target_password="ABCD", candidate_words=["ABCD", "EFGH", "IJKL", "MNOP", "QRST"]
    )

    # 4 attempts allowed
    game.make_guess("EFGH")  # 3 left
    game.make_guess("IJKL")  # 2 left
    game.make_guess("MNOP")  # 1 left
    result = game.make_guess("QRST")  # 0 left

    assert "Lockdown initiated" in result
    assert not game.has_won
    assert game.is_game_over
    assert game.attempts_left == 0

    # A vs N
    # Only 1 match (S) at index 0
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


def test_game_invalid_guess() -> None:
    """Test making an invalid guess (not in candidate list)."""
    candidates = ["WORD1", "WORD2"]
    game = Game(target_password="WORD1", candidate_words=candidates)
    result = game.make_guess("INVALID")
    assert result == "Entry denied."
    assert game.attempts_left == 4  # No penalty
    assert not game.is_game_over


def test_game_correct_guess() -> None:
    """Test making a correct guess."""
    candidates = ["WORD1", "WORD2"]
    game = Game(target_password="WORD1", candidate_words=candidates)
    result = game.make_guess("WORD1")
    assert result == "Correct."
    assert game.has_won
    assert game.is_game_over


def test_game_incorrect_guess() -> None:
    """Test making an incorrect guess."""
    candidates = ["WORD1", "WORD2"]
    game = Game(target_password="WORD1", candidate_words=candidates)
    # WORD2 matches WORD1 in 4 chars: W, O, R, D. But wait, WORD1 vs WORD2?
    # W-W, O-O, R-R, D-D, 1-2 (No).
    # So likeness is 4.
    result = game.make_guess("WORD2")
    assert "Likeness=4" in result
    assert game.attempts_left == 3
    assert not game.is_game_over
    assert game.history == [("WORD2", 4)]


def test_game_over_attempts() -> None:
    """Test losing the game by running out of attempts."""
    candidates = ["WORD1", "WORD2"]
    game = Game(target_password="WORD1", candidate_words=candidates, attempts_left=1)
    result = game.make_guess("WORD2")
    assert "Lockdown initiated" in result
    assert game.attempts_left == 0
    assert game.is_game_over
    assert not game.has_won
