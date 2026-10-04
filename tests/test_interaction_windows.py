import runpy
from pathlib import Path

API = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/collect_interaction_windows.py"))


def window(*lines):
    return API["label"](next(API["chains"](list(lines))))


def test_target_departure_uses_owner_and_pre_resolution_order():
    row = window("Message chaining,", "1 Activating h4 (Veiler)",
                 "0 Agent targets m2 (Raye)", "Message chaining,",
                 "0 Activating m2 (Raye)", "0 You tribute m2 (Raye).",
                 "Message chain_solving,", "Message chain_end,")
    assert row["labels"]["target_left_before_first_resolution"] is True
    assert row["escape_evidence_lines"] == [[3, 6]]
    assert row["labels"]["opponent_target_left_before_first_resolution"] is True
    assert row["labels"]["own_target_left_before_first_resolution"] is None
    assert row["labels"]["effect_success"] is None


def test_other_owner_or_late_movement_does_not_prove_escape():
    for actor, early in ((1, True), (0, False)):
        events = ["Message chaining,", "1 Activating h4 (Veiler)", "0 Agent targets m2 (Raye)"]
        if not early:
            events.append("Message chain_solving,")
        events += [f"{actor} You tribute m2 (Raye).", "Message chain_end,"]
        assert window(*events)["labels"]["target_left_before_first_resolution"] is None


def test_disabled_is_not_attributed_to_last_source():
    row = window("Message chaining,", "0 Activating g1 (Fusion)",
                 "Message chaining,", "1 Activating h2 (Ash)",
                 "1 You discarded h2 (Ash)", "Message chain_solving,",
                 "Message chain_disabled,", "Message chain_end,")
    assert row["labels"]["revealed_discard_then_disable_message"] is True
    assert row["labels"]["effect_success"] is None
    assert "disabled_source" not in row


def test_self_target_departure_is_not_opponent_escape():
    row = window("Message chaining,", "1 Activating s3 (Multirole)",
                 "0 Bob targets om3 (Raye)", "Message chaining,",
                 "1 Activating m3 (Raye)", "1 You tribute m3 (Raye).",
                 "Message chain_solving,", "Message chain_end,")
    assert row["labels"]["own_target_left_before_first_resolution"] is True
    assert row["labels"]["opponent_target_left_before_first_resolution"] is None


def test_hidden_discard_and_incomplete_chain_are_not_exported():
    row = window("Message chaining,", "0 Activating s1 (Twin)",
                 "0 You discarded h1 (HiddenCard)", "Message chain_end,")
    assert "HiddenCard" not in str(row)
    assert list(API["chains"](["Message chaining,", "0 Activating s1 (Twin)"])) == []


def test_same_seed_and_combo_cannot_cross_fitted_partitions():
    rows = [(seed, combo, API["partition"](seed, combo))
            for seed in range(2026104400, 2026104432) for combo in ("A", "B", "C", "D")]
    for field in (0, 1):
        sets = [{r[field] for r in rows if r[2] == split} for split in ("train", "validation", "test")]
        assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert API["partition"](2026104421, "A") == "development_root"
