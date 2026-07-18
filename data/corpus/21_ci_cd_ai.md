# CI for AI Engineering

Run pytest on every PR including RAG and agent smoke tests.
Cache pip wheels; avoid downloading large embedding models in CI.
Fail the build if golden-set citation rate drops below a threshold.
Optional nightly job can call a real model behind encrypted secrets.
