# Fallout Terminal Minigame Implementation Plan

## Overview
The goal is to turn the current static grid display into an interactive terminal hacking minigame. The player must guess a password from a list of words displayed on the screen. Feedback is given in the form of "likeness" (number of characters in the correct position).

## User Experience
1.  **Start**: The user runs the program. A grid of characters with embedded words is displayed (similar to current output).
2.  **Loop**:
    -   The user is prompted to enter a guess.
    -   If the guess is the correct word, they win.
    -   If the guess is one of the other displayed words, the system updates the "attempts remaining" and displays the "likeness" score.
    -   If the guess is invalid (not in list), an error is shown (attempts are not penalized).
    -   The game ends when the user wins or runs out of attempts (usually 4).
3.  **End**: Display a success or failure message.

## Architecture

### 1. `Game` Class (New)
A class to manage the game state.

**State:**
-   `target_password`: `str` - The correct word.
-   `candidate_words`: `list[str]` - All valid words on the screen.
-   `attempts_left`: `int` - Counter (starts at 4).
-   `history`: `list[tuple[str, int]]` - Log of (guess, likeness).
-   `is_game_over`: `bool`.
-   `has_won`: `bool`.

**Methods:**
-   `__init__(target: str, candidates: list[str], attempts: int = 4)`
-   `make_guess(word: str) -> str`:
    -   Validates input (case-insensitive, must be in `candidate_words`).
    -   Calculates likeness.
    -   Updates attempts and history.
    -   Returns a feedback string (e.g., "Entry denied", "Likeness=2", "Correct").
-   `calculate_likeness(word1: str, word2: str) -> int`:
    -   Static/Utility method.
    -   Counts matching characters at the same index.

### 2. Refactor `__main__.py`
-   Move grid generation logic into a setup function that returns the game context (the words and the rendered grid).
-   Implement the main game loop using the `Game` class.
-   Current `main()` only prints. Changed to `run_game()`.

### 3. Display Logic
-   Retain the existing `grid.py` logic for visual flair.
-   Current printing in `__main__.py` produces the visual output. We should keep this.

## Implementation Steps

1.  **Create `game.py`**:
    -   Implement `Game` class.
    -   Implement `calculate_likeness`.

2.  **Add Tests**:
    -   Test `calculate_likeness` logic.
    -   Test `Game` state transitions (correct guess, incorrect guess, game over).

3.  **Update `__main__.py`**:
    -   Select a `target_password` randomly from the generated `chosen_words`.
    -   Initialize `Game`.
    -   Enter `input()` loop.

## Testing Strategy
-   **Unit Tests**:
    -   `test_likeness`: Verify reasoning (e.g., "SCORPION" vs "SCORPION" -> 8, "SCORPION" vs "SARGIOAA" -> matches S, R, I, O? No, strict index matching).
    -   `test_game_flow`: Ensure attempts decrement, game ends on 0 attempts or correct guess.
