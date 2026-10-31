# Call Me Maybe
_This project has been created as part of the 42 curriculum by zhewu._

42 project of LLM function calling

We use a state-based constrained decoder to force valid JSON tokens. As a deterministic fallback for rare structural timeouts, we employ a structural repair mechanism that balances braces and removes trailing commas before parsing.

# Resources

https://docs.python.org/ja/3/library/json.html
