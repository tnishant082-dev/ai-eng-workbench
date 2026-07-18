# Get Time Tool Spec

`get_time()` returns ISO-8601 local time and timezone label for the workbench host.
Deterministic in tests via dependency injection or freezegun-style stubs.
Demonstrates a side-effect-free tool with clear observability.
Include timezone in the string so demos in IST read correctly.
