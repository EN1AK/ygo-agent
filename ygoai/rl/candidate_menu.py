"""Fail-closed identity for the policy-visible staged action menu.

This does not invent unstaged card combinations. A captured slot always refers
to exactly the row the actor was offered at that decision.
"""

import hashlib
from dataclasses import dataclass
from typing import Mapping, NamedTuple

import numpy as np


MENU_ID_VERSION = 1
ACTION_FIELDS = (
    "actions_", "action_features_", "action_single_refs_",
    "action_group_refs_", "action_group_mask_",
)


class CandidateMenuError(ValueError):
    pass


@dataclass(frozen=True)
class CandidateMenu:
    version: int
    digest: str
    action_digests: tuple[str, ...]
    valid_mask: np.ndarray
    num_options: int
    chosen_index: int


class CandidateMenuBatch(NamedTuple):
    """Numeric rollout sidecar; digest bytes survive JAX sharding unchanged."""

    version: np.ndarray
    digest: np.ndarray
    valid_mask: np.ndarray
    num_options: np.ndarray
    chosen_index: np.ndarray


def _update_array(digest, array):
    value = np.ascontiguousarray(array)
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes())


def capture_candidate_menu(
    observation: Mapping[str, np.ndarray], num_options: int, chosen_index: int,
) -> CandidateMenu:
    actions = np.asarray(observation["actions_"])
    if actions.ndim != 2 or actions.shape[1] < 4:
        raise CandidateMenuError("actions_ is not one unbatched staged menu")
    capacity = actions.shape[0]
    count = int(num_options)
    selected = int(chosen_index)
    if not 1 <= count <= capacity:
        raise CandidateMenuError(f"num_options {count} outside 1..{capacity}")
    if not 0 <= selected < count:
        raise CandidateMenuError(f"chosen index {selected} outside menu of {count}")
    actor_valid = actions[:, 3] != 0
    actor_valid[0] = True  # The actor always unmasks the first slot.
    expected = np.arange(capacity) < count
    if not np.array_equal(actor_valid, expected):
        raise CandidateMenuError("num_options disagrees with the actor's action mask")
    fields = {}
    for name in ACTION_FIELDS:
        if name not in observation:
            if name == "actions_":
                raise CandidateMenuError("missing actor action rows")
            continue
        value = np.asarray(observation[name])
        if value.shape[0] != capacity:
            raise CandidateMenuError(f"{name} has a different action capacity")
        fields[name] = value
    if any(name in fields for name in ACTION_FIELDS[1:]) and not all(
        name in fields for name in ACTION_FIELDS[1:]
    ):
        raise CandidateMenuError("incomplete structured action identity")

    identities = []
    for index in range(count):
        digest = hashlib.blake2b(digest_size=16)
        digest.update(f"staged-action-v{MENU_ID_VERSION}".encode("ascii"))
        for name in ACTION_FIELDS:
            if name in fields:
                digest.update(name.encode("ascii"))
                _update_array(digest, fields[name][index])
        identities.append(digest.hexdigest())
    if len(set(identities)) != count:
        raise CandidateMenuError("duplicate policy-visible action identity")

    digest = hashlib.blake2b(digest_size=32)
    digest.update(f"staged-menu-v{MENU_ID_VERSION}".encode("ascii"))
    for name in ("global_", "selection_"):
        if name in observation:
            digest.update(name.encode("ascii"))
            _update_array(digest, observation[name])
    digest.update(np.asarray([count, capacity], dtype="<i8").tobytes())
    for identity in identities:
        digest.update(bytes.fromhex(identity))
    return CandidateMenu(
        MENU_ID_VERSION, digest.hexdigest(), tuple(identities), expected,
        count, selected)


def verify_candidate_menu(
    captured: CandidateMenu, observation: Mapping[str, np.ndarray],
    num_options: int, chosen_index: int,
) -> None:
    current = capture_candidate_menu(observation, num_options, chosen_index)
    if (
        captured.version != current.version
        or captured.digest != current.digest
        or captured.action_digests != current.action_digests
        or captured.num_options != current.num_options
        or captured.chosen_index != current.chosen_index
        or not np.array_equal(captured.valid_mask, current.valid_mask)
    ):
        raise CandidateMenuError("staged menu identity or chosen slot changed")


def capture_candidate_menu_batch(
    observations: Mapping[str, np.ndarray], num_options, chosen_index,
) -> CandidateMenuBatch:
    """Capture the same per-environment menu the actor used for its action."""
    counts = np.asarray(num_options)
    choices = np.asarray(chosen_index)
    actions = np.asarray(observations["actions_"])
    if actions.ndim != 3 or counts.shape != (actions.shape[0],) or choices.shape != counts.shape:
        raise CandidateMenuError("batched menu, counts, and chosen indices disagree")
    menus = []
    for env_index in range(actions.shape[0]):
        observation = {
            name: np.asarray(value)[env_index]
            for name, value in observations.items()
            if value is not None and (name in ACTION_FIELDS or name in ("global_", "selection_"))
        }
        try:
            menus.append(capture_candidate_menu(
                observation, int(counts[env_index]), int(choices[env_index])))
        except CandidateMenuError as exc:
            raise CandidateMenuError(f"environment {env_index}: {exc}") from exc
    return CandidateMenuBatch(
        version=np.asarray([menu.version for menu in menus], dtype=np.uint8),
        digest=np.asarray([list(bytes.fromhex(menu.digest)) for menu in menus], dtype=np.uint8),
        valid_mask=np.stack([menu.valid_mask for menu in menus]),
        num_options=counts.astype(np.int32),
        chosen_index=choices.astype(np.int32),
    )


def verify_candidate_menu_batch(
    captured: CandidateMenuBatch, observations: Mapping[str, np.ndarray],
    num_options, chosen_index,
) -> None:
    current = capture_candidate_menu_batch(observations, num_options, chosen_index)
    for field in CandidateMenuBatch._fields:
        if not np.array_equal(getattr(captured, field), getattr(current, field)):
            raise CandidateMenuError(f"batched staged menu {field} changed")
