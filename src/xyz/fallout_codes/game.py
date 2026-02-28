"""Game logic for the Fallout-style code-breaking minigame."""

from dataclasses import dataclass, field


@dataclass
class Game:
    """Manages the state of the game."""

    target_password: str
    candidate_words: list[str]
    attempts_left: int = 4
    history: list[tuple[str, int]] = field(default_factory=list)
    is_game_over: bool = False
    has_won: bool = False

    def __post_init__(self) -> None:
        """Validate game initialization."""
        self.candidate_words = [w.upper() for w in self.candidate_words]
        self.target_password = self.target_password.upper()
        if self.target_password not in self.candidate_words:
            # It's possible the game logic allows the target to be secret, but in this
            # minigame style, the target is usually one of the displayed options.
            # We will allow it but maybe warn or just proceed.
            pass

    def make_guess(self, word: str) -> str:
        """Process a user's guess."""
        word = word.upper().strip()

        if self.is_game_over:
            return "Game over."

        if word not in self.candidate_words:
            return "Entry denied."  # Not a valid word choice

        likeness = calculate_likeness(word, self.target_password)
        self.history.append((word, likeness))

        if word == self.target_password:
            self.has_won = True
            self.is_game_over = True
            return "Correct."

        self.attempts_left -= 1
        if self.attempts_left <= 0:
            self.is_game_over = True
            return f"Likeness={likeness}. Lockdown initiated."

        return f"Likeness={likeness}."


def calculate_likeness(word1: str, word2: str) -> int:
    """Calculates the number of matching characters at the same index."""
    count = 0
    # Words are assumed to be same length in this game, but let's be safe
    for c1, c2 in zip(word1, word2):
        if c1 == c2:
            count += 1
    return count
