import os
import sqlite3
from glob import glob

from dataclasses import dataclass
import tyro

@dataclass
class Args:
    output: str = "code_list.txt"
    """the file containing the list of card codes"""
    cdb: str = "../assets/locale/en/cards.cdb"
    """the cards database file"""
    script_dir: str = "script"
    """path to the scripts directory"""


def read_card_codes(cdb):
    with sqlite3.connect(f"file:{cdb}?mode=ro", uri=True) as conn:
        return sorted(row[0] for row in conn.execute("SELECT id FROM datas"))

if __name__ == "__main__":
    args = tyro.cli(Args)

    pattern = os.path.join(args.script_dir, "c*.lua")
    # list all c*.lua files
    script_files = glob(pattern)

    codes = sorted([os.path.basename(f).split(".")[0][1:] for f in script_files])
    codes_s = set()
    for code in codes:
        try:
            codes_s.add(int(code))
        except ValueError:
            pass
    codes_c = read_card_codes(args.cdb)
    codes_c_set = set(codes_c)

    scripts_missing_in_cdb = codes_s.difference(codes_c_set)
    if len(scripts_missing_in_cdb) > 0:
        raise ValueError(f"Missing in cards.cdb: {sorted(scripts_missing_in_cdb)[:20]}")

    cards_missing_scripts = codes_c_set.difference(codes_s)
    if len(cards_missing_scripts) > 0:
        print(f"Cards without scripts: {len(cards_missing_scripts)}")

    print(f"Total {len(codes_c)} cards, {len(codes_s)} scripts")

    lines = []
    for c in codes_c:
        line = f"{c} {1 if c in codes_s else 0}"
        lines.append(line)
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines))
