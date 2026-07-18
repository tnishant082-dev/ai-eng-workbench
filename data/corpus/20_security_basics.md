# Security Basics for LLM Apps

Never log raw API keys; redact Authorization headers in traces.
Validate tool arguments to block path traversal and shell injection.
Treat model output as untrusted when rendering HTML in UIs.
Corpus files are user-controlled input — sanitize before display.
