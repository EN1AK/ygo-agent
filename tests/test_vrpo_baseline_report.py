import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.eval_vrpo_baseline import main, parse_episodes, validate_first_episodes, validate_replay_decisions
from ygoai.rl.checkpoint_compat import sha256_file
from ygoai.rl.match_report import summarize_matches


def line(number, reward, invalid=0, reason=1):
    return (f"Episode {number}: length=12, reward={reward}, "
            f"win={int(float(reward) > 0)}, reason=1, "
            f"invalid_game={invalid}, termination_reason={reason}\n")


class BaselineReportTest(unittest.TestCase):
    def test_natural_outcomes_and_timeouts_are_distinct(self):
        rows = parse_episodes(line(1, 1) + line(2, -1) + line(3, 0)
                              + line(4, 3, reason=2) + line(5, 1, invalid=1))
        report = summarize_matches(rows)
        self.assertEqual(report["valid_games"], 3)
        self.assertEqual(report["invalid_games"], 2)
        self.assertEqual([report[k] for k in ("wins", "losses", "draws")], [1, 1, 1])

    def test_nonfinite_and_inconsistent_records_fail_closed(self):
        for text in (line(1, "nan"), line(1, "inf"), line(2, 1),
                     line(1, 1) * 2, line(1, 1).replace("win=1", "win=0")):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_episodes(text)


    def test_first_episode_pairing_rejects_duplicate_or_missing_environment(self):
        text = line(1, 1).rstrip() + ", environment_index=1\n"
        text += line(2, -1).rstrip() + ", environment_index=0\n"
        rows = parse_episodes(text)
        validate_first_episodes(rows, 2)
        self.assertEqual([row["environment_index"] for row in rows], [1, 0])
        for bad in (rows[:1], [rows[0], rows[0]], parse_episodes(line(1, 1) + line(2, -1))):
            with self.assertRaises(ValueError):
                validate_first_episodes(bad, 2)

    def test_replay_requires_complete_finite_unassisted_decisions_and_terminal(self):
        decision = dict(record_type="decision", step=0, player=1, checkpoint_sha256="model",
                        legal_actions=[{"index": 0}], policy_logits=[0.], policy_probabilities=[1.],
                        state_value=0.5, selected_action=0, raw_selected_action=0, cycle_guard_intervened=False)
        terminal = dict(record_type="terminal", steps=1, checkpoint_sha256="model",
                        terminal_reward=1., invalid_game=0, termination_reason=1)
        episode = dict(reward=1., invalid_game=0, termination_reason=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.jsonl"
            def write(rows):
                path.write_text("\n".join(json.dumps(row) for row in rows))
            write([decision, terminal])
            self.assertTrue(validate_replay_decisions(path, "model", 1, episode)["complete_raw_policy_log"])
            with self.assertRaisesRegex(ValueError, "raw win reason"):
                validate_replay_decisions(path, "model", 1, {**episode, "win_reason": 2})
            for broken in (
                [decision], [decision, {**terminal, "steps": 2}],
                [{**decision, "state_value": float("nan")}, terminal],
                [{**decision, "policy_probabilities": [0.5]}, terminal],
                [{**decision, "cycle_guard_intervened": True}, terminal],
                [{**decision, "step": 1}, terminal],
                [{**decision, "checkpoint_sha256": "wrong"}, terminal],
            ):
                write(broken)
                with self.assertRaises(ValueError):
                    validate_replay_decisions(path, "model", 1, episode)

    def test_manifest_uses_paired_batches_and_isolated_replay_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("repo/scripts/script/procedure.lua", "repo/assets/locale/zh/cards.cdb",
                         "model", "model.metadata.json", "train.log", "codes", "semantics/metadata.json",
                         "native.so", "decks/elfnote.ydk"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
            argv = ["eval_vrpo_baseline"]
            for flag, name in (("repo", "repo"), ("checkpoint", "model"), ("training-log", "train.log"),
                               ("decks", "decks"), ("code-list", "codes"), ("semantics", "semantics"),
                               ("native-module", "native.so"), ("output", "output")):
                argv += [f"--{flag}", str(root / name)]
            argv += ["--source-commit", "fixture", "--deck-names", "elfnote", "--seeds", "7",
                     "--episodes-per-cell", "2"]
            checkpoint_hash = sha256_file(root / "model")
            def evaluator(command, *, cwd, stdout, **kwargs):
                if "--record" not in command:
                    self.assertIn("--first-episode-per-env", command)
                    self.assertEqual(command[command.index("--num-envs") + 1], "2")
                    stdout.write((line(1, 1).rstrip() + ", environment_index=1\n" +
                                  line(2, -1).rstrip() + ", environment_index=0\n").encode())
                else:
                    self.assertNotEqual(Path(cwd), root / "repo")
                    replay = Path(cwd) / "replay"
                    replay.mkdir()
                    (replay / "fixture.yrp").write_bytes(b"fixture-only")
                    stdout.write(line(1, 1).encode())
                    seat = int(command[command.index("--player") + 1])
                    rows = [dict(record_type="decision", step=0, player=seat,
                                 checkpoint_sha256=checkpoint_hash, legal_actions=[{"index": 0}],
                                 policy_logits=[0.], policy_probabilities=[1.], state_value=0.,
                                 selected_action=0, raw_selected_action=0, cycle_guard_intervened=False),
                            dict(record_type="terminal", steps=1, checkpoint_sha256=checkpoint_hash,
                                 terminal_reward=1., invalid_game=0, termination_reason=1)]
                    Path(command[command.index("--decision-log") + 1]).write_text(
                        "\n".join(json.dumps(row) for row in rows))
                return SimpleNamespace(returncode=0)
            with patch("sys.argv", argv), patch("subprocess.run", side_effect=evaluator) as calls:
                main()
            manifest = json.loads((root / "output/baseline-manifest.json").read_text())
            self.assertEqual(calls.call_count, 4)  # two seat batches and two replays, no duplicated deck
            self.assertEqual(manifest["status"], "evidence_complete_manual_metrics_pending")
            self.assertEqual(len(manifest["cells"]), 2)
            self.assertEqual(len(manifest["replays"]), 2)
            self.assertTrue(all(r["decision_validation"]["complete_raw_policy_log"] for r in manifest["replays"]))
            self.assertTrue(all(r["commentary"] is None for r in manifest["replays"]))


if __name__ == "__main__":
    unittest.main()
