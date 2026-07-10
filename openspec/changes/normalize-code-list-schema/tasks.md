## 1. Shared Parser

- [x] 1.1 Add shared Python code-list parsing helpers that accept single-column and metadata-column lines.
- [x] 1.2 Add parser validation behavior for blank lines, malformed card codes, and optional `has_script` metadata.

## 2. Python Tooling

- [x] 2.1 Update embedding loading in `ygoai.utils` to use the shared parser.
- [x] 2.2 Update `scripts/card/embedding.py` to read existing code lists through the shared parser and preserve append behavior.
- [x] 2.3 Ensure generated or appended code-list entries remain deterministic and compatible with the selected output format.

## 3. Native Environment Readers

- [x] 3.1 Update YGOPro-v0 native code-list reader to parse the first whitespace-separated column.
- [x] 3.2 Update EDOPro native code-list reader to parse the first whitespace-separated column.
- [x] 3.3 Keep YGOPro-v1 two-column `has_script` handling intact.

## 4. Documentation And Verification

- [x] 4.1 Update README documentation for the generated `<card_code> <has_script>` format and update command.
- [x] 4.2 Add or run verification that current `scripts/code_list.txt` parses in Python tooling.
- [x] 4.3 Run Python compile checks for modified scripts and modules.
- [x] 4.4 Validate the OpenSpec change with `openspec validate normalize-code-list-schema --strict`.
