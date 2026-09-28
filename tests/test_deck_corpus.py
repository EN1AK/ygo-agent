import json
import tempfile
import unittest
import zipfile
import subprocess
from pathlib import Path

from ygoai.deck_corpus import (
    CorpusError,
    assign_families,
    build_corpus,
    freeze_corpus_revision,
    materialize_source,
    materialize_canonical_decks,
    parse_ydk_text,
    validate_artifact,
    validate_canonical_deck,
    validate_registry,
)


DECK_A = """#created by test
#main
1
2
2
#extra
9
!side
7
"""


class DeckCorpusTest(unittest.TestCase):
    def validation_fixture(self):
        deck = {
            "canonical_id": "deck-test", "zone_sizes": {"main": 40, "extra": 0, "side": 0},
            "zones": {"main": {"1": 40}, "extra": {}, "side": {}},
        }
        return deck, {1: {"alias": 0, "type": 0}}, {1: 1}, {1: 0}, {1}

    def test_static_validation_reports_each_failure_class(self):
        with tempfile.TemporaryDirectory() as temp:
            script_dir = Path(temp)
            deck, database, positions, scripts, semantics = self.validation_fixture()
            self.assertEqual(validate_canonical_deck(
                deck, database, positions, scripts, semantics, script_dir)["status"], "ready")

            cases = []
            bad_size = json.loads(json.dumps(deck))
            bad_size["zone_sizes"]["main"] = 39
            cases.append((bad_size, database, positions, scripts, semantics,
                          "invalid_deck_size", "excluded"))
            cases.append((deck, {}, positions, scripts, semantics,
                          "missing_card_database_row", "excluded"))
            cases.append((deck, {1: {"alias": 0, "type": 0x4000}}, positions, scripts,
                          semantics, "token_card_in_deck", "excluded"))
            bad_zone = json.loads(json.dumps(deck))
            bad_zone["zones"] = {"main": {}, "extra": {"1": 1}, "side": {}}
            bad_zone["zone_sizes"] = {"main": 40, "extra": 1, "side": 0}
            cases.append((bad_zone, database, positions, scripts, semantics,
                          "invalid_zone_card_type", "excluded"))
            cases.append((deck, database, {}, {}, set(),
                          "missing_code_list_entry", "limited"))
            cases.append((deck, database, positions, {1: 1}, semantics,
                          "missing_required_script", "limited"))
            cases.append((deck, database, positions, scripts, set(),
                          "missing_semantic_row", "limited"))
            for candidate, db, pos, script_flags, known, reason, status in cases:
                result = validate_canonical_deck(
                    candidate, db, pos, script_flags, known, script_dir)
                self.assertEqual(result["status"], status, reason)
                self.assertIn(reason, [item["code"] for item in result["reasons"]])

    def test_materialized_canonical_deck_is_stable(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "decks"
            canonical = {"decks": [{
                "canonical_id": "deck-test",
                "zones": {"main": {"2": 2, "1": 1}, "extra": {"9": 1}, "side": {"7": 1}},
            }]}
            first = materialize_canonical_decks(canonical, ["deck-test"], output)
            second = materialize_canonical_decks(canonical, ["deck-test"], output)
            self.assertEqual(first, second)
            self.assertEqual(
                (output / "deck-test.ydk").read_text(),
                "#created by ygo-agent deck-corpus-v1\n#main\n1\n2\n2\n#extra\n9\n!side\n7\n",
            )

    def test_freeze_only_samples_ready_and_preserves_aliases(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            deck = {
                "canonical_id": "deck-ready", "canonical_hash": "h", "family_id": "f",
                "zone_sizes": {"main": 40, "extra": 0, "side": 0},
                "zones": {"main": {"1": 40}, "extra": {}, "side": {}},
                "aliases": [{"source_id": "s", "source_path": "a.ydk", "name": "a",
                             "source_sha256": "x"}],
            }
            excluded = {**deck, "canonical_id": "deck-excluded", "canonical_hash": "x"}
            documents = {
                "canonical.json": {"decks": [deck, excluded]},
                "static.json": {"decks": [
                    {"canonical_id": "deck-ready", "status": "ready", "reasons": []},
                    {"canonical_id": "deck-excluded", "status": "excluded", "reasons": [
                        {"code": "invalid", "severity": "excluded", "card_codes": [], "detail": "x"}]},
                ]},
                "runtime.json": {"decks": {"deck-ready": {"status": "passed"}}},
                "source.json": {}, "families.json": {}, "rejected.json": {"decks": [{"source_path": "bad"}]},
            }
            for name, document in documents.items():
                (root / name).write_text(json.dumps(document))
            manifest = freeze_corpus_revision(
                root / "canonical.json", root / "static.json", root / "runtime.json",
                root / "source.json", root / "families.json", root / "rejected.json", root / "v1",
            )
            self.assertEqual(manifest["counts"], {"ready": 1, "limited": 0, "excluded": 1})
            sampling = json.loads((root / "v1/sampling-input.json").read_text())
            self.assertEqual([item["canonical_id"] for item in sampling["decks"]], ["deck-ready"])
            ready = json.loads((root / "v1/ready.json").read_text())
            self.assertEqual(ready["decks"][0]["deck"]["aliases"][0]["source_path"], "a.ydk")

    def test_parser_is_order_independent(self):
        left = parse_ydk_text(DECK_A)
        right = parse_ydk_text("#main\n2\n1\n2\n#extra\n9\n!side\n7\n")
        self.assertEqual(left.canonical_hash(), right.canonical_hash())

    def test_parser_rejects_malformed_cards(self):
        with self.assertRaises(CorpusError):
            parse_ydk_text("#main\nnot-a-card\n")

    def test_registry_requires_provenance(self):
        with self.assertRaises(CorpusError):
            validate_registry({"schema_version": 1, "acquired_at": "x", "sources": [{"id": "x"}]})

    def test_all_artifact_contracts_accept_valid_and_reject_missing_fields(self):
        fixtures = {
            "canonicalDeck": {"canonical_id": "d", "canonical_hash": "h", "zones": {}, "zone_sizes": {}, "aliases": [], "family_id": "f"},
            "validation": {"canonical_id": "d", "status": "ready", "reasons": []},
            "family": {"family_id": "f", "members": ["d"]},
            "feature": {"schema_version": 1, "matrix_hash": "h", "row_ids": [], "column_ids": []},
            "cluster": {"cluster_id": "c", "representative": "d", "members": ["d"], "metrics": {}},
            "split": {"revision": "r", "train_families": [], "held_out_families": []},
            "corpusManifest": {"revision": "r", "source_registry_hash": "a", "canonical_decks_hash": "b", "validation_hash": "c", "family_hash": "d"},
        }
        for kind, fixture in fixtures.items():
            validate_artifact(kind, fixture)
            invalid = dict(fixture)
            invalid.pop(next(iter(invalid)))
            with self.assertRaises(CorpusError, msg=kind):
                validate_artifact(kind, invalid)

    def test_exact_dedup_and_alias_preservation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            decks = root / "decks"
            decks.mkdir()
            (decks / "a.ydk").write_text(DECK_A)
            (decks / "b.ydk").write_text(DECK_A)
            registry = {
                "schema_version": 1,
                "acquired_at": "2026-09-23T00:00:00+08:00",
                "sources": [{
                    "id": "fixture", "kind": "local", "location": "decks",
                    "revision": "fixture-v1", "license": "test", "provenance": "test",
                    "include": ["*.ydk"], "exclude": [],
                }],
            }
            registry_path = root / "registry.json"
            registry_path.write_text(json.dumps(registry))
            report = build_corpus(registry_path, root, root / "out", root / "cache")
            self.assertEqual(report["raw_file_count"], 2)
            self.assertEqual(report["canonical_deck_count"], 1)
            canonical = json.loads((root / "out/canonical-decks.json").read_text())
            self.assertEqual(len(canonical["decks"][0]["aliases"]), 2)

    def test_near_duplicate_family(self):
        def item(name, cards):
            return {"canonical_id": name, "zones": {"main": cards, "extra": {}, "side": {}}}
        families = assign_families([
            item("a", {"1": 20, "2": 20}),
            item("b", {"1": 20, "2": 19, "3": 1}),
            item("c", {"8": 20, "9": 20}),
        ], 0.90)
        self.assertEqual(sorted(len(x["members"]) for x in families), [1, 2])

    def test_archive_adapter_is_pinned_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "fixture.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("decks/a.ydk", DECK_A)
            import hashlib
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            source = {
                "id": "archive", "kind": "archive", "location": "fixture.zip",
                "revision": digest,
            }
            first = materialize_source(source, root, root / "cache")
            second = materialize_source(source, root, root / "cache")
            self.assertEqual((first / "decks/a.ydk").read_bytes(), (second / "decks/a.ydk").read_bytes())

    def test_git_adapter_is_pinned_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Deck Corpus Test"], cwd=repo, check=True)
            (repo / "a.ydk").write_text(DECK_A)
            subprocess.run(["git", "add", "a.ydk"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                text=True, capture_output=True,
            ).stdout.strip()
            source = {
                "id": "git", "kind": "git", "location": str(repo),
                "revision": revision,
            }
            first = materialize_source(source, root, root / "cache")
            second = materialize_source(source, root, root / "cache")
            self.assertEqual((first / "a.ydk").read_bytes(), (second / "a.ydk").read_bytes())


if __name__ == "__main__":
    unittest.main()
