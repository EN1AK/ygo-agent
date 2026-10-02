"""Build a hash-pinned, semantics-preserving fusion helper patch in a new file.

This changes the order in which ALREADY SELECTED cards are assigned to material
roles, not which cards can be selected and not the engine response protocol.
Runtime installation and its backup/manifest are deliberately separate.
"""
import argparse
import hashlib
import json
from pathlib import Path


UPSTREAM_SHA256 = "64e510643688d192e1b2ad4c979e131df93401a519fbcf6493bdf5d42b8ebd3c"
REPLACEMENTS = (
    ("\t\treturn sg:IsExists(Auxiliary.FCheckMixRepSelected,1,g,tp,mg,sg,g,...)\n",
     "\t\t-- Assign every selected card once in a fixed order; still try every role.\n"
     "\t\tlocal remaining=sg:Clone()\n"
     "\t\tremaining:Sub(g)\n"
     "\t\tlocal nextcard=remaining:GetFirst()\n"
     "\t\treturn nextcard~=nil and Auxiliary.FCheckMixRepSelected(nextcard,tp,mg,sg,g,...)\n"),
    ("\t\tres=sg:IsExists(Auxiliary.FCheckMixRepSelected,1,nil,tp,mg,sg,g,fc,sub,chkfnf,...)\n",
     "\t\t-- The candidate has already been added to sg, which is nonempty.\n"
     "\t\tlocal ordered=sg:Clone()\n"
     "\t\tres=Auxiliary.FCheckMixRepSelected(ordered:GetFirst(),tp,mg,sg,g,fc,sub,chkfnf,...)\n"),
)


def patch_procedure(raw: bytes) -> bytes:
    if hashlib.sha256(raw).hexdigest() != UPSTREAM_SHA256:
        raise ValueError("Unknown procedure.lua: audit the new upstream before patching")
    text = raw.decode("utf-8").replace("\r\n", "\n")
    for old, new in REPLACEMENTS:
        if text.count(old) != 1:
            raise ValueError("Fusion helper anchor is missing or not unique")
        text = text.replace(old, new)
    return text.encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite an existing procedure or patch artifact")
    patched = patch_procedure(args.input.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(patched)
    print(json.dumps({
        "patch": "fusion-selected-role-order-v1", "input_sha256": UPSTREAM_SHA256,
        "output_sha256": hashlib.sha256(patched).hexdigest(),
        "output": str(args.output.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
