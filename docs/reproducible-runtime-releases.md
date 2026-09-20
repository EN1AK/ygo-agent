# Reproducible runtime releases

Every native environment/model version used for training must be frozen as one
immutable release before a server run starts. A source checkout by itself is
not a runnable release because Python can silently load an older extension and
online package recipes can change independently of the repository.

Each release must retain:

- the exact Git commit and a clean/dirty status record;
- the Python version, dependency lock/export, compiler and build-tool versions;
- the native `ygoenv` extension and its SHA-256;
- both observation-schema manifests and a real-engine schema self-test result;
- semantic assets plus their metadata and source hashes;
- the code list, deck/assets revision, launch command and resolved arguments;
- every starting checkpoint, sidecar and SHA-256.

Generated models and native binaries stay outside Git. Put the release under
`dist/runtime-releases/<release-id>/`, generate `manifest.json` with
`scripts/create_runtime_manifest.py`, and copy the complete directory to the
server. Do not rebuild an already named release in place. If any artifact
changes, create a new release ID.

Before training, the server must verify every manifest hash and run both the
legacy and requested schema self-tests. A matching filename is not evidence of
compatibility. Training must stop if the extension omits a required tensor,
the schema manifest differs, or a checkpoint sidecar does not match.

Large checkpoints and `.so` files must not be committed to GitHub. Keep them in
the local `dist` archive and on the training server; Git contains the manifest,
build instructions, reports and small configuration files only.
