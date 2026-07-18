# OpenAI-Compatible APIs

Many providers expose `/v1/chat/completions` with the OpenAI request shape.
Set `OPENAI_BASE_URL` and `OPENAI_API_KEY` to swap providers without code changes.
Never commit keys; use `.env` locally and secret managers in real deploys.
Token estimates can use `len(text)/4` when the provider does not return usage.
