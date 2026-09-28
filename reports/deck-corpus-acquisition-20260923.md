# Deck Corpus Acquisition Report (2026-09-23)

## Outcome

The first clustered multi-deck corpus acquisition is complete. It is an
acquisition/canonicalization result, not yet a runtime-eligibility result: native
load, script/code-list coverage, and smoke-duel gates remain in OpenSpec section 3.

## Pinned sources

| Source | Revision | License note | YDK files |
|---|---|---|---:|
| ygo-agent local decks | `9148ec0ce37ec942b29a4eec09e5e6bad5c8c454` | repository license | 37 |
| anonfansub/ygodecks | `950dc0f2452e5e79c48398efc607e6fdd94c4215` | `NOASSERTION` (no license file) | 781 |
| isaiasgv/yugi-decks | `328d0c0390aabd6b8e944e93f186bb08178d1410` | `NOASSERTION` (no license file) | 52 |
| larikk/ygo-ydk-files | `a773da9d9b5a409fcca9f44e542acacfdc637c54` | `NOASSERTION` (no license file) | 77 |
| ProjectIgnis/WindBot-Ignite | `3539531b381316d1e045040c899bfa295a77a9f2` | AGPL-3.0-or-later | 42 |

External sources without a declared license are retained for private research
with explicit `NOASSERTION` provenance; this report does not grant redistribution
rights.

## Counts

- Raw YDK files: **989**
- Successfully parsed: **989**
- Parse rejects: **0**
- Exact canonical decks: **911**
- Exact duplicate aliases removed from sampling identity: **78**
- Near-duplicate families at weighted-Jaccard threshold 0.90: **862**
- Elfnote canonical ID: `deck-03a44c5091635d1e`
- Elfnote family ID: `family-37a555621313f6fa`

The target of at least 256 candidate canonical decks is met. The word `ready`
must not be applied until the section 3 runtime validation completes.

## Reproducibility

- Source registry SHA-256: `fdb3dca893dcb4875957189867800c0d00a141e8e3117c896f17622e78bb4c24`
- Canonical decks SHA-256: `afc53d32e96e9cd5adbd6c65b25d99936690cf3e201251d44f38c69128fd12e5`
- Families SHA-256: `0c386a99e61830a933d16472f2ed267abdf33789bcd60c730883decbcf5db945`
- Source manifest SHA-256: `32d6faad9a845291577eb7d80e3dd62b972a67e3311db69ad47c64d8cca7a569`
- Report SHA-256: `d24ae10114faaf10c11b57ffdf9e723779918044b0432f8ae52cfcde459d6700`

A clean second build from the same pinned sources produced identical hashes for
the source manifest, canonical deck table, family table, rejection table, and
summary report.

## 40M parent trace

The parent manifest verifies the 40M checkpoint SHA-256
`a6ddf4b08d5dba2cbe786b3df8b465cc4b01d8d8b85104187a98cb166a14496f`,
Structured-lite/full metadata, semantic metadata hash, code-list hash, elfnote
hash, repository revision, and compute manifest.

The current checkout stores `code_list.txt` and `elfnote.ydk` with LF line
endings. The 40M metadata/report hashes were produced from content-identical CRLF
files. Exact CRLF parent bytes were therefore frozen under
`corpus/frozen-parent-assets/`; this avoids mistaking newline conversion for a
card-order or deck-content migration.

## Artifact location

`training-runs/deck-corpus-raw-20260923/`

- `sources/`: pinned git working copies
- `corpus/raw/`: source-separated acquired YDK files
- `corpus/source-manifest.json`: per-source provenance and every file hash
- `corpus/canonical-decks.json`: canonical zones, aliases, hashes, and families
- `corpus/families.json`: near-duplicate family assignments
- `corpus/rejected.json`: parsing rejects
- `corpus/report.json`: machine-readable acquisition totals
- `corpus/parent-manifest.json`: 40M continuation parent trace
- `corpus/compute-manifest.json`: hardware/runtime trace
- `corpus-rebuild-check/`: deterministic second build
