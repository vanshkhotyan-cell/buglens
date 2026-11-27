# Developer Setup Guide

1. Activate environment (conda activate buglens) and sync locked packages (uv sync --locked).
2. Install git hooks: pre-commit install.
3. Copy .env.example to .env (gitignored) and set your DagsHub token.
4. Configure local DVC credentials: python -m dvc remote modify origin --local access_key_id <TOKEN> and secret_access_key <TOKEN>.
5. Pull datasets: python -m dvc pull.
