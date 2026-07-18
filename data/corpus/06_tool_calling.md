# Tool Calling Agents

A minimal agent loop: plan → call tool → observe → decide next step or final answer.
Keep tool schemas small (name, description, JSON args) and validate arguments.
Always emit trace events for tool_start, tool_end, and model_call for debugging.
Prefer deterministic tools (calculator, search, get_time) before adding network tools.
