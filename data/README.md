# Data handling

- `smoke_cases.jsonl` contains harmless prompts for end-to-end software checks only.
- Add benchmark and preconstructed research cases to a private, git-ignored path unless redistribution is permitted.
- Keep a frozen development/test split and document any benchmark transformations.
- `behavior` is passed to the HarmBench judge. A benign smoke behavior is not a meaningful HarmBench ASR task; ignore its score for research claims.
- Do not include secrets, credentials, or personal data in conversations or logs.
