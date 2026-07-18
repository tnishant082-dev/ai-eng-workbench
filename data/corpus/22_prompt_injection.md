# Prompt Injection Awareness

Retrieved docs and user text can instruct the model to ignore system rules.
Tool-using agents amplify risk because injected text may trigger tool calls.
Mitigations: allow-listed tools, strict schemas, and separate system channels.
Evals should include at least one injection-style adversarial case.
