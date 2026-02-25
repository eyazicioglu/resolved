# Conventions

## Agent Output Style
- no emojis. no filler phrases. no prose. no redundant statements.
- dense, minimal, clean. minimal but sufficient — do 80% right, don't obsess over the remaining 20%.
- use tables and lists over paragraphs.
- applies to all agent-generated text including session log entries.

## Code Architecture

### Functions vs Classes
- stateless logic as functions. classes only when managing shared resources (model, db, connection pool).
- avoid monolithic classes that accumulate unrelated responsibilities — split early.

### File Size
- soft limit: ~400 lines per file.
- if a file exceeds this or handles multiple distinct responsibilities, split it.
- agents must not produce monolithic files. when in doubt, split.

### Module Structure
- one concern per file, grouped in directories.
- example:
  ```
  services/
  ├── search.py        # search logic only
  ├── scoring.py       # scoring/normalization only
  ├── embedding.py     # model loading, encoding only
  ├── filtering.py     # filter logic only
  ```

### Shared Utilities
- reusable helpers go in `utils/`.
- only extract into `utils/` when shared by 2+ modules. otherwise keep helpers local.

## Code Style

### Comments
- lowercase. uppercase only for abbreviations (e.g., FAISS, API, WSI).
- no trailing periods.
- minimal — code should be self-documenting. comment only non-obvious logic.

### Type Hints
- enforced on all function signatures.
  ```python
  def search(query: str, top_k: int = 20) -> list[dict]:
  ```

### Docstrings
- only for non-obvious functions. one-liner format.
  ```python
  def encode_text(query: str) -> np.ndarray:
      """encode query text into 512-dim vector"""
  ```

### Imports
- relative within the same package, absolute for everything else.
  ```python
  # absolute for external and cross-package
  import numpy as np
  from app.services.search import SearchEngine

  # relative within same package
  from .scoring import stretch_score
  ```

## Config
- config file (YAML/JSON) for project settings.
- `.env` strictly for secrets and environment-specific values.
- always provide a `.env.example`.

## Error Handling
- raise exceptions — don't return error codes.
- `ValueError` for bad input, `RuntimeError` for system failures.
- never silently swallow exceptions — always log at minimum.

## Testing
- every new function/method needs a unit test.
- test file naming: `test_{module}.py`.
- test data goes in `tests/fixtures/` — never use production data.

## Git
- commit prefixes: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`
- branches: `{type}/{short-name}` (e.g., `feat/search-history`). delete after merge.
- never commit: `.env`, `data/`, `__pycache__/`, `node_modules/`

## Folder Structure (locked)
Do not create new top-level directories without explicit approval. Current structure:

```
project/
├── AGENTS.md              # agent instructions (read first)
├── CONVENTIONS.md         # this file
├── README.md
├── docs/
│   ├── specs/             # feature specs (one .md per feature)
│   ├── SESSION_LOG.md     # agent session memory
│   └── KNOWN_ISSUES.md   # resolved bugs, open debt
├── src/ or app/           # source code (project-dependent)
├── tests/                 # test suite
│   └── fixtures/          # synthetic test data
├── utils/                 # shared helpers (only when needed)
├── config/                # config files
├── scripts/               # tooling, automation
└── data/                  # runtime data (gitignored)
```

New feature specs go in `docs/specs/{FEATURE_NAME}.md`.

