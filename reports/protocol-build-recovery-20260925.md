# Protocol native build recovery — 2026-09-25

## Root cause

The GPU server has GLIBC 2.35 and xmake `2.9.9+20260911`, but its installed
xmake script directory had a newer `utils.replace` rule that calls
`on_prepare_file`, which this executable does not provide. The v3.0.9/v3.1.1
Linux bundles provide that rule/API but require GLIBC 2.38, so they cannot run
on this server.

The server's cached `gflags` and `glog` static libraries were also built against
GLIBC 2.38 (`__isoc23_strto*` undefined symbols on this host).

## Isolated recovery

- Kept all work under `/tmp/ygo-proto-build.pa9KjH`; did not replace the
  server-wide xmake installation or shared package cache.
- Downloaded the official xmake v2.9.9 source scripts from tag commit
  `40815a0e2ff8790fce9e9bf7e4c3ec3d866571cf` and set
  `XMAKE_PROGRAM_DIR=/tmp/ygo-proto-build.pa9KjH/xmake-program-v2.9.9` to match
  the existing executable.
- Set `XMAKE_GLOBALDIR=/tmp/ygo-proto-build.pa9KjH/xmake-global`; populated only
  that private cache with the server's cached package files and the locally
  uploaded missing SQLite 3.43 source archive.
- Rebuilt `gflags 2.3.1` and `glog 0.6.0` from the server's cached source archives
  using its GCC 11.4 / CMake 3.22.1, then replaced only their libraries in the
  private package copy. Neither archive needed another upload.
- Cleared and regenerated the isolated project xmake config, then built
  `ygopro_ygoenv` from the current source snapshot with native optimization off
  (the portable default).

## Verification

- Build: passed; extension SHA-256
  `f8fcd5d066b1acb1575ea4809f782828eb635277441d70fbcb9de2b9efd603a5`.
- Runtime import resolved to the isolated extension and all expected
  `_protocol_*` bindings were present.
- `python -m unittest discover -s tests -p 'test_ygocore_protocol*.py' -v`:
  21 tests passed.
- The first run found an assertion typo expecting `24 legal actions` when the
  fixture correctly reports `25 legal actions`; the assertion now checks the
  actual count and model capacity.
- No training was started. This is a helper/boundary suite result, not evidence
  that every parser branch has yet passed a real pinned-core response oracle.
