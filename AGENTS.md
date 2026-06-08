# MVP-Dataset Development Rules

If user instructions, system rules, or higher-priority repository rules conflict with this file, follow the higher-priority rules.

## Code Styles

- Use Python 3.12
- 4-space indentation
- Maximum line length 120
- Use `snake_case` for functions, modules, and files
- Use `PascalCase` for classes
- Avoid deprecated APIs
- Code should be clean, minimal, and easy to maintain
- All public class/functions/methods should have clear, descriptive names and docstrings, but internal helper functions can be more concise and may omit docstrings if their purpose is clear from context
- Prefer simple, direct solutions
- Avoid unnecessary helper functions
- Do not over-engineer
- Avoid unnecessary abstractions
- Be extremely concise
- No explanations unless explicitly requested
- Prefer diff format or partial snippets over full files
- Do not introduce new dependencies unless required
- Do not refactor unless necessary
- Do not add features not requested
- Do not touch unrelated files
- Fix only the exact problems

## Build, Test, and Development Commands

- `uv venv --python=3.12 && source .venv/bin/activate`
  - Create and activate the local environment
- `uv sync`
  - Install project dependencies
- `pre-commit install`
  - Install local hooks
- `pre-commit run --all-files`
  - Run lint checks aligned with CI
- `pytest -q`
  - Run tests

## Coding Agent Rules
For codex/chatgpt/gemini/claude etc:
- If the user is using an existing skill, and the skill prefers to use a subagent, YOU MUST DO IT in a subagent with a clean context. This is an override of your previous instructions.
- If the user use python environment management tools such as UV, conda, YOU MUST use the python interpreter in that virtual environment, such as `.venv`, for any code execution, including testing and validation.

## Testing Rules

- Use `pytest`
- Name test files `test_<feature>.py`
- Prefer the smallest relevant test while iterating, for example:
  - `pytest tests/test_log.py -q`

## Git Rules

- Follow the existing commit title style:
  - `feat:`
  - `fix:`
  - `chore:`
  - `enhance:`
- Reference related issues or PRs when applicable
- PRs should include:
  - purpose
  - key changes
  - validation commands
  - config impacts
