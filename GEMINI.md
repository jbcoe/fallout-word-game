# Gemini User Preferences

This file summarizes the development preferences for this user.

- **Testing:** Tests should be run using `uv run pytest`.
- **Error Handling:** For command-line applications, errors during setup or execution should raise exceptions that are _not_ caught by the main entry point. The program should terminate and display a full stack trace. Do not implement graceful exits unless explicitly asked.
- **Code Style:** Follow existing code idioms and patterns within the project. When in doubt, prioritize consistency with the current codebase over external suggestions.
- **Test Design:** Test names should make failure meaningful. Tests should meaningfully test a code path, with meaningfully different input when tests are parameterized.
