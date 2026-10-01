import itertools
import unittest

from ygoenv.ygopro import ygopro_ygoenv as native


def core_sum_check(params, index, acc, opmin=0xFFFF):
    if acc == 0 or index >= len(params):
        return False
    first = params[index] & 0xFFFF
    second = params[index] >> 16
    if index == len(params) - 1:
        return ((acc == first and acc + opmin > first) or
                (second != 0 and acc == second and acc + opmin > second))
    return ((acc > first and core_sum_check(
                params, index + 1, acc - first, min(first, opmin))) or
            (second > 0 and acc > second and core_sum_check(
                params, index + 1, acc - second, min(second, opmin))))


def core_sum_limit_check(params, acc):
    if not params:
        return False
    minimum_values = []
    maximum_values = []
    for param in params:
        first = param & 0xFFFF
        second = param >> 16
        minimum_values.append(second if second and second < first else first)
        maximum_values.append(max(first, second))
    return sum(maximum_values) >= acc and sum(minimum_values) - min(minimum_values) < acc


class ProtocolBoundaryTest(unittest.TestCase):
    @staticmethod
    def _u32(value):
        return list(int(value).to_bytes(4, "little"))

    def test_cancel_is_distinct_in_legacy_action_history(self):
        # MSG_SELECT_TRIBUTE maps to a nonzero message row; Cancel is act row 9.
        encoded = native._protocol_cancel_history_encoding_fixture(20)
        self.assertEqual(len(encoded), 14)
        self.assertNotEqual(encoded[3], 0)
        self.assertEqual(encoded[4], 9)

    def test_hidden_card_code_zero_keeps_unknown_identity(self):
        self.assertEqual(native._protocol_unknown_card_placeholder(), (0, 0))

    def test_greedy_baseline_uses_semantic_priority(self):
        self.assertEqual(native._greedy_action_fixture(), [1, 1, 1])

    def test_battle_and_idle_commands_through_adapter_and_same_core(self):
        for battle, expected in ((True, 4), (False, 9)):
            for index, command in enumerate(
                    ([0, 1, 2, 3] if battle else range(9))):
                frame, actions, response, accepted = (
                    native._protocol_command_adapter_fixture(battle, index))
                self.assertEqual(frame[0], 10 if battle else 11)
                self.assertEqual(actions, expected)
                self.assertEqual(response & 0xFFFF, command)
                self.assertEqual(response >> 16, 0)
                self.assertTrue(accepted)

    def test_chain_forced_optional_and_zero_through_adapter_and_same_core(self):
        for count, forced, index, choices, response in (
                (0, False, -1, 0, -1),
                (1, False, 0, 2, 0),
                (1, False, 1, 2, -1),
                (2, False, 1, 3, 1),
                (2, False, 2, 3, -1),
                (1, True, 0, 1, 0),
                (2, True, 1, 2, 1)):
            frame, actions, actual, accepted = (
                native._protocol_chain_adapter_fixture(count, forced, index))
            self.assertEqual(frame[0], 16)
            self.assertEqual(actions, choices)
            self.assertEqual(actual, response)
            self.assertTrue(accepted)

    def test_announce_card_rpn_and_arbitrary_code_through_same_core(self):
        for named in (False, True):
            for index in range(2):
                frame, actions, response, accepted = (
                    native._protocol_announce_card_adapter_fixture(
                        named, index))
                self.assertEqual(frame[0], 142)
                self.assertEqual(actions, 2)
                expected = (23456790, 23456789)[index] if named else (
                    23456789, 23456790)[index]
                self.assertEqual(response, expected)
                self.assertTrue(accepted)

    def test_large_announce_card_is_staged_without_losing_any_code(self):
        for decisions, expected_sizes, expected_code in (
                ([0, 0, 0], [16, 10, 2], 23456789),
                ([15, 14], [16, 15], 23457088)):
            frame, sizes, response, accepted = (
                native._protocol_announce_card_large_adapter_fixture(
                    decisions))
            self.assertEqual(frame[0], 142)
            self.assertEqual(sizes, expected_sizes)
            self.assertEqual(response, expected_code)
            self.assertTrue(accepted)

    def test_command_response_preserves_command_and_full_uint16_index(self):
        response = native._protocol_command_response(5, 0xFFFF)
        self.assertEqual(response, 0xFFFF0005)
        self.assertEqual(response & 0xFFFF, 5)
        self.assertEqual(response >> 16, 0xFFFF)

    def test_action_capacity_fails_closed_instead_of_truncating_legal_actions(self):
        native._protocol_validate_action_capacity(24, 24)
        with self.assertRaisesRegex(RuntimeError, "25 legal actions exceed model capacity 24"):
            native._protocol_validate_action_capacity(25, 24)

    def test_new_policy_messages_append_only_within_frozen_embedding(self):
        self.assertEqual(native._protocol_message_id(11), 1)
        self.assertEqual(native._protocol_message_id(25), 18)
        self.assertEqual(native._protocol_message_id(22), 19)
        self.assertEqual(native._protocol_message_id(132), 20)

    def test_num_options_spec_accepts_exact_capacity(self):
        defaults = dict(zip(
            native._YGOProEnvSpec._config_keys,
            native._YGOProEnvSpec._default_config_values,
        ))
        spec = native._YGOProEnvSpec(
            native._YGOProEnvSpec._default_config_values)
        state = dict(zip(spec._state_keys, spec._state_spec))
        self.assertEqual(
            state["info:num_options"][2], (0, defaults["max_options"]))

    def test_select_unselect_response_indices_cover_both_card_pools(self):
        self.assertEqual(
            native._protocol_select_unselect_response_index(3, 2, False, 2), 2)
        self.assertEqual(
            native._protocol_select_unselect_response_index(3, 2, True, 1), 4)
        with self.assertRaises(RuntimeError):
            native._protocol_select_unselect_response_index(3, 2, True, 2)

    def test_policy_hides_navigation_cancel_but_keeps_semantic_decline(self):
        for message in (15, 20, 26):
            before, after, has_cancel, suppressed = (
                native._protocol_policy_cancel_filter_fixture(message))
            self.assertEqual((before, after), (2, 1))
            self.assertFalse(has_cancel)
            self.assertTrue(suppressed)

        before, after, has_cancel, suppressed = (
            native._protocol_policy_cancel_filter_fixture(16))
        self.assertEqual((before, after), (2, 2))
        self.assertTrue(has_cancel)
        self.assertFalse(suppressed)

    def test_policy_blocks_all_unselect_when_forward_choice_exists(self):
        fixture = native._protocol_policy_unselect_filter_fixture
        self.assertEqual(fixture(True, False, False),
                         (3, 1, False, False, 0))
        self.assertEqual(fixture(False, False, False),
                         (2, 2, True, True, -1))
        self.assertEqual(fixture(True, True, False),
                         (3, 3, True, True, 0))

    def test_policy_suppresses_unselect_only_when_finish_is_available(self):
        fixture = native._protocol_policy_unselect_filter_fixture
        self.assertEqual(fixture(True, False, True),
                         (4, 2, False, False, 0))
        self.assertEqual(fixture(False, False, True),
                         (3, 1, False, False, -1))
        self.assertEqual(fixture(True, True, True),
                         (4, 4, True, True, 0))

    def test_select_unselect_pinned_core_request_and_index_validator(self):
        for index in range(4):
            request, accepted, result = native._protocol_select_unselect_core_fixture(
                2, 2, False, False, index)
            self.assertEqual(request[:7], [26, 0, 0, 0, 1, 1, 2])
            self.assertEqual(request[23], 2)
            self.assertEqual(len(request), 40)
            self.assertEqual(int.from_bytes(bytes(request[7:11]), "little"), 1000)
            self.assertEqual(int.from_bytes(bytes(request[24:28]), "little"), 2000)
            self.assertTrue(accepted)
            self.assertEqual(result, [])
        _, accepted, result = native._protocol_select_unselect_core_fixture(
            2, 2, False, False, 4)
        self.assertFalse(accepted)
        self.assertEqual(result, [1])

    def test_select_unselect_termination_forms_match_core(self):
        for finishable, cancelable in ((True, False), (False, True), (True, True)):
            request, accepted, result = native._protocol_select_unselect_core_fixture(
                1, 1, finishable, cancelable, -1)
            self.assertEqual(request[:4], [26, 0, int(finishable), int(cancelable)])
            self.assertTrue(accepted)
            self.assertEqual(result, [])
        _, accepted, result = native._protocol_select_unselect_core_fixture(
            1, 1, False, False, -1)
        self.assertFalse(accepted)
        self.assertEqual(result, [1])

    def test_select_unselect_frame_through_actual_adapter_and_same_core(self):
        for index in range(4):
            frame, actions, accepted = (
                native._protocol_select_unselect_adapter_fixture(
                    2, 2, False, False, index))
            self.assertEqual(frame[0], 26)
            self.assertEqual(actions, 4)
            self.assertTrue(accepted)
        for finishable, cancelable in ((True, False), (False, True)):
            frame, actions, accepted = (
                native._protocol_select_unselect_adapter_fixture(
                    1, 1, finishable, cancelable, 2))
            self.assertEqual(actions, 3)
            self.assertTrue(accepted)

    def test_select_unselect_uint8_index_boundary(self):
        index = native._protocol_select_unselect_response_index(255, 1, True, 0)
        self.assertEqual(index, 255)
        _, accepted, result = native._protocol_select_unselect_core_fixture(
            255, 1, False, False, index)
        self.assertTrue(accepted)
        self.assertEqual(result, [])
        with self.assertRaises(RuntimeError):
            native._protocol_select_unselect_response_index(255, 2, True, 1)

    def test_announced_number_preserves_full_uint32_option_value(self):
        self.assertEqual(native._protocol_uint32_identity(0xFFFFFFFF), 0xFFFFFFFF)

    def test_announced_numbers_and_position_through_adapter_and_core(self):
        numbers = [17, 0x80000001, 0xFFFFFFFF]
        for index in range(len(numbers)):
            frame, actions, response, accepted = (
                native._protocol_number_adapter_fixture(numbers, index))
            self.assertEqual(frame[:3], [143, 0, 3])
            self.assertEqual(actions, 3)
            self.assertEqual(response, index)
            self.assertTrue(accepted)
        frame, actions, response, accepted = (
            native._protocol_position_adapter_fixture(0b0110, 1))
        self.assertEqual(frame[0], 19)
        self.assertEqual(actions, 2)
        self.assertEqual(response, 4)
        self.assertTrue(accepted)

    def test_scalar_yes_no_option_chain_and_rps_responses(self):
        self.assertEqual(native._protocol_scalar_response(0, 0, 1), 0)
        self.assertEqual(native._protocol_scalar_response(1, 0, 1), 1)
        self.assertEqual(native._protocol_scalar_response(17, 0, 17), 17)
        self.assertEqual(
            list(native._protocol_rock_paper_scissors_options()), [1, 2, 3])
        for hand in native._protocol_rock_paper_scissors_options():
            self.assertEqual(native._protocol_scalar_response(hand, 1, 3), hand)
        self.assertEqual(native._protocol_scalar_response(-1, -1, 8), -1)
        self.assertEqual(native._protocol_scalar_response(3, 1, 3), 3)
        with self.assertRaises(RuntimeError):
            native._protocol_scalar_response(4, 1, 3)

    def test_yesno_effectyn_and_option_frames_through_adapter_and_core(self):
        for kind, message in ((0, 13), (1, 12)):
            for action_index, expected in ((0, 1), (1, 0)):
                frame, actions, response, accepted = (
                    native._protocol_scalar_adapter_fixture(kind, action_index))
                self.assertEqual(frame[0], message)
                self.assertEqual(actions, 2)
                self.assertEqual(response, expected)
                self.assertTrue(accepted)
        for action_index in range(3):
            frame, actions, response, accepted = (
                native._protocol_scalar_adapter_fixture(2, action_index))
            self.assertEqual(frame[:3], [14, 0, 3])
            self.assertEqual(actions, 3)
            self.assertEqual(response, action_index)
            self.assertTrue(accepted)
        frame, actions, response, accepted = (
            native._protocol_scalar_adapter_fixture(3, 0))
        self.assertEqual(frame[:3], [14, 0, 1])
        self.assertEqual(actions, 1)
        self.assertEqual(response, 0)
        self.assertTrue(accepted)

    def test_rps_pinned_core_emits_both_requests_and_validates_all_responses(self):
        for player in (0, 1):
            self.assertEqual(
                native._protocol_rock_paper_scissors_player(player), player)
        with self.assertRaisesRegex(RuntimeError, "Invalid MSG_ROCK_PAPER_SCISSORS player"):
            native._protocol_rock_paper_scissors_player(2)
        for invalid_hand in (0, 4):
            with self.assertRaisesRegex(RuntimeError, "outside the declared range"):
                native._protocol_rock_paper_scissors_core_fixture(
                    [invalid_hand, 1], False)

        for first, second in itertools.product((1, 2, 3), repeat=2):
            frames, winner = native._protocol_rock_paper_scissors_core_fixture(
                [first, second], False)
            self.assertEqual(frames, [[132, 0], [132, 1], [133, first + (second << 2)]])
            expected = (2 if first == second else
                        1 if (first, second) in ((1, 2), (2, 3), (3, 1))
                        else 0)
            self.assertEqual(winner, expected)

    def test_rps_core_repeat_draw_requests_both_players_again(self):
        frames, winner = native._protocol_rock_paper_scissors_core_fixture(
            [2, 2, 1, 2], True)
        self.assertEqual(
            frames,
            [[132, 0], [132, 1], [133, 10], [132, 0], [132, 1], [133, 9]],
        )
        self.assertEqual(winner, 1)

    def test_rps_frames_through_actual_adapter_and_same_core(self):
        for first, second in itertools.product((1, 2, 3), repeat=2):
            frames, winner = native._protocol_rps_adapter_fixture(
                [first, second], False)
            self.assertEqual(frames, [
                [132, 0], [132, 1], [133, first + (second << 2)]])
            self.assertIn(winner, (0, 1, 2))
        frames, winner = native._protocol_rps_adapter_fixture(
            [2, 2, 1, 2], True)
        self.assertEqual(frames, [
            [132, 0], [132, 1], [133, 10],
            [132, 0], [132, 1], [133, 9]])
        self.assertEqual(winner, 1)

    def test_core_emitted_notifications_parse_exactly_and_preserve_boundaries(self):
        u32 = self._u32
        reload_field = [162, 5]
        for _ in range(2):
            reload_field += u32(8000)
            reload_field += [0] * 7   # empty monster zones
            reload_field += [0] * 8   # empty spell/trap zones
            reload_field += [0] * 6   # deck/hand/grave/banished/extra/face-up extra
        reload_field += [0]           # no active chain

        frames = [
            [35, 0],                                      # swap deck/grave
            [42, 0, 1] + u32(1000) + [0, 0x40, 0],       # confirm extra top
            [97] + u32(0x00000400) + u32(0x00000401),    # cancel target
            [133, 1 + (2 << 2)],                         # RPS result
            [161, 0, 1, 1, 0, 1] + u32(1000) +
                u32(1001) + u32(1002),                   # tag swap
            reload_field,
            [163, 2, 0, ord("A"), ord("I"), 0],          # AI name
            [164, 4, 0, ord("h"), ord("i"), ord("n"),
             ord("t"), 0],                              # show hint
            [170] + u32(1003),                           # match kill
        ]
        expected = [35, 42, 97, 133, 161, 162, 163, 164, 170]
        concatenated = [byte for frame in frames for byte in frame]
        self.assertEqual(
            native._protocol_notification_adapter_fixture(concatenated),
            expected,
        )

        # This exact shape caused the 2.97M pilot crash: two players each swap
        # deck/grave and immediately receive the following shuffle notification.
        observed = [35, 0, 32, 0, 35, 1, 32, 1]
        self.assertEqual(
            native._protocol_notification_adapter_fixture(observed),
            [35, 32, 35, 32],
        )

    def test_notification_parser_rejects_truncated_payload(self):
        with self.assertRaisesRegex(RuntimeError, "Truncated message confirm_extratop"):
            native._protocol_notification_adapter_fixture([42, 0, 1])

    def test_optional_empty_and_index_responses(self):
        self.assertEqual(native._protocol_index_response([], 4), [0])
        self.assertEqual(native._protocol_index_response([3, 0], 4), [2, 3, 0])
        with self.assertRaises(RuntimeError):
            native._protocol_index_response([1, 1], 4)
        with self.assertRaises(RuntimeError):
            native._protocol_index_response([4], 4)

    def test_select_card_optional_and_cancel_through_adapter_and_core(self):
        for minimum, maximum, cancelable, decisions, expected_sizes in (
                (0, 2, False, [3], [4]),
                (1, 2, True, [3], [4]),
                (1, 2, True, [0, 2], [4, 4])):
            frame, sizes, accepted = native._protocol_select_card_adapter_fixture(
                3, minimum, maximum, cancelable, decisions)
            self.assertEqual(frame[:5], [15, 0, int(cancelable), minimum, maximum])
            self.assertEqual(sizes, expected_sizes)
            self.assertTrue(accepted)

    def test_exact_sum_sequences_match_core_validator(self):
        must = [2]
        optional = [3, 5, 4 | (6 << 16)]
        sequences = native._protocol_sum_sequences(must, optional, 9, 1, 2)
        self.assertTrue(sequences)
        for sequence in sequences:
            params = must + [optional[index] for index in sequence]
            self.assertTrue(core_sum_check(params, 0, 9))
            self.assertGreaterEqual(len(sequence), 1)
            self.assertLessEqual(len(sequence), 2)

    def test_exact_sum_sequences_are_unique_card_sets_not_permutations(self):
        optional = [1] * 14
        sequences = native._protocol_sum_sequences([], optional, 7, 7, 7)
        self.assertEqual(len(sequences), 3432)
        self.assertEqual(
            len({tuple(sorted(sequence)) for sequence in sequences}),
            len(sequences),
        )
        for sequence in sequences:
            self.assertEqual(sequence, sorted(sequence))
            self.assertTrue(core_sum_check(
                [optional[index] for index in sequence], 0, 7))

    def test_exact_sum_zero_value_is_rejected_like_core(self):
        optional = [2, 0]
        sequences = native._protocol_sum_sequences([], optional, 2, 2, 2)
        self.assertEqual(sequences, [])

    def test_sum_limit_combinations_are_complete(self):
        must = [2]
        optional = [3, 5, 4 | (6 << 16)]
        actual = {
            tuple(combination)
            for combination in native._protocol_sum_limit_combinations(
                must, optional, 8)
        }
        expected = set()
        for size in range(len(optional) + 1):
            for combination in itertools.combinations(range(len(optional)), size):
                params = must + [optional[index] for index in combination]
                if core_sum_limit_check(params, 8):
                    expected.add(combination)
        self.assertEqual(actual, expected)

    def test_tribute_uses_core_count_and_weight_rules(self):
        actual = {
            tuple(combination)
            for combination in native._protocol_tribute_combinations(
                [2, 1, 3], 3, 2)
        }
        self.assertEqual(actual, {(0, 1), (0, 2), (2,), (1, 2)})
        self.assertIn(
            [], native._protocol_tribute_combinations([1, 2], 0, 2))

    def test_staged_weighted_choices_match_exhaustive_small_oracle(self):
        cases = [
            (0, [], [2, 1, 3], 3, 0, 2),
            (1, [2], [3, 5, 4 | (6 << 16)], 9, 1, 2),
            (2, [2], [3, 5, 4 | (6 << 16)], 8, 0, 3),
        ]
        for kind, must, optional, target, minimum, maximum in cases:
            if kind == 0:
                complete = native._protocol_tribute_combinations(
                    optional, target, maximum)
            elif kind == 1:
                complete = native._protocol_sum_sequences(
                    must, optional, target, minimum, maximum)
            else:
                complete = native._protocol_sum_limit_combinations(
                    must, optional, target)
            prefixes = {tuple()}
            for combination in complete:
                for length in range(len(combination) + 1):
                    prefixes.add(tuple(combination[:length]))
            for prefix in prefixes:
                choices, finishable = native._protocol_weighted_staged_choices(
                    kind, must, optional, list(prefix), target,
                    minimum, maximum, 100000)
                expected_choices = sorted({
                    combination[len(prefix)] for combination in complete
                    if len(combination) > len(prefix)
                    and tuple(combination[:len(prefix)]) == prefix
                })
                self.assertEqual(choices, expected_choices)
                self.assertEqual(finishable, list(prefix) in complete)

    def test_large_weighted_prompt_is_incremental_and_budgeted(self):
        optional = [1] * 30
        choices, finishable = native._protocol_weighted_staged_choices(
            1, [], optional, [], 15, 15, 15, 100000)
        self.assertEqual(choices, list(range(16)))
        self.assertFalse(finishable)
        completed = set()
        for pick_last in (False, True):
            selected = []
            for _ in range(15):
                choices, finishable = native._protocol_weighted_staged_choices(
                    1, [], optional, selected, 15, 15, 15, 100000)
                self.assertTrue(choices)
                self.assertFalse(finishable)
                selected.append(choices[-1] if pick_last else choices[0])
            choices, finishable = native._protocol_weighted_staged_choices(
                1, [], optional, selected, 15, 15, 15, 100000)
            self.assertEqual(choices, [])
            self.assertTrue(finishable)
            self.assertEqual(selected, sorted(set(selected)))
            _, accepted, result = native._protocol_weighted_core_fixture(
                1, [], optional, 15, 15, 15, selected)
            self.assertTrue(accepted)
            self.assertEqual(result, [])
            completed.add(tuple(selected))
        self.assertEqual(len(completed), 2)
        with self.assertRaisesRegex(RuntimeError, "search budget exhausted"):
            native._protocol_weighted_staged_choices(
                1, [], [1] * 28, [], 29, 0, 28, 100)

    def test_weighted_responses_pass_pinned_core(self):
        cases = [
            (0, [], [2, 1, 3], 3, 0, 2),
            (1, [2], [3, 5, 4 | (6 << 16)], 9, 1, 2),
            (2, [2], [3, 5, 4 | (6 << 16)], 8, 0, 3),
        ]
        for kind, must, optional, target, minimum, maximum in cases:
            if kind == 0:
                combinations = native._protocol_tribute_combinations(
                    optional, target, maximum)
            elif kind == 1:
                combinations = native._protocol_sum_sequences(
                    must, optional, target, minimum, maximum)
            else:
                combinations = native._protocol_sum_limit_combinations(
                    must, optional, target)
            self.assertTrue(combinations)
            for selected in combinations:
                request, accepted, result = native._protocol_weighted_core_fixture(
                    kind, must, optional, target, minimum, maximum, selected)
                self.assertEqual(request[0], 20 if kind == 0 else 23)
                self.assertTrue(accepted, (kind, selected, result))
                self.assertEqual(result, [])

    def test_weighted_frames_through_actual_adapter_and_same_core_instance(self):
        cases = [
            (0, [], [2, 1, 3], 3, 0, 2, [2, 0], [3, 1]),
            (1, [2], [3, 5, 4 | (6 << 16)], 9, 1, 2,
             [0, 0, 0], [1, 1, 1]),
            (2, [2], [3, 5, 4 | (6 << 16)], 8, 0, 3,
             [1, 0], [2, 1]),
        ]
        for (kind, must, optional, target, minimum, maximum,
             decisions, expected_sizes) in cases:
            frame, branch_sizes, accepted = native._protocol_weighted_adapter_fixture(
                kind, must, optional, target, minimum, maximum, decisions)
            self.assertEqual(frame[0], 20 if kind == 0 else 23)
            self.assertEqual(branch_sizes, expected_sizes)
            self.assertTrue(accepted)

    def test_tribute_cancel_zero_min_and_core_max_cap(self):
        frame, sizes, accepted = native._protocol_tribute_adapter_fixture(
            [2, 1, 3], 2, 2, True, [3])
        self.assertEqual(frame[:5], [20, 0, 1, 2, 2])
        self.assertEqual(sizes, [4])
        self.assertTrue(accepted)

        frame, sizes, accepted = native._protocol_tribute_adapter_fixture(
            [2, 1, 3], 0, 2, False, [3])
        self.assertEqual(frame[:5], [20, 0, 0, 0, 2])
        self.assertEqual(sizes, [4])
        self.assertTrue(accepted)

        frame, sizes, accepted = native._protocol_tribute_adapter_fixture(
            [1] * 6, 1, 7, False, [0, 5])
        self.assertEqual(frame[4], 5)
        self.assertEqual(sizes, [6, 6])
        self.assertTrue(accepted)

    def test_counter_allocation_is_exact_and_capacity_bounded(self):
        self.assertEqual(
            native._protocol_counter_allocation([1, 5, 2], 6),
            [1, 5, 0],
        )
        with self.assertRaises(RuntimeError):
            native._protocol_counter_allocation([1, 2], 4)

    def test_counter_allocation_bytes_pass_pinned_core(self):
        for allocation in ([0, 4, 2], [1, 3, 2], [1, 5, 0]):
            request, accepted, result = native._protocol_select_counter_core_fixture(
                [1, 5, 2], 6, allocation)
            self.assertEqual(request[0], 22)
            self.assertEqual(request[6], 3)
            self.assertTrue(accepted)
            self.assertEqual(result, [])
        _, accepted, result = native._protocol_select_counter_core_fixture(
            [1, 5, 2], 6, [1, 5, 1])
        self.assertFalse(accepted)
        self.assertEqual(result, [1])

    def test_counter_frame_through_actual_adapter_and_same_core_instance(self):
        for decisions, expected in (
                ([0, 0], [0, 4, 2]),
                ([1, 0, 0], [1, 3, 2]),
                ([1, 1], [1, 5, 0])):
            frame, branch_sizes, actual, accepted = (
                native._protocol_counter_adapter_fixture(
                    [1, 5, 2], 6, decisions))
            self.assertEqual(frame[:7], [22, 0, 1, 0, 6, 0, 3])
            self.assertEqual(branch_sizes, [2] * len(decisions))
            self.assertEqual(actual, expected)
            self.assertTrue(accepted)
        for capacities, requested, decisions, expected in (
                ([1], 1, [], [1]),
                ([1, 1], 1, [0], [0, 1]),
                ([1, 1], 1, [1], [1, 0])):
            _, branch_sizes, actual, accepted = (
                native._protocol_counter_adapter_fixture(
                    capacities, requested, decisions))
            self.assertEqual(branch_sizes, [2] * len(decisions))
            self.assertEqual(actual, expected)
            self.assertTrue(accepted)

    def test_sort_staged_order_and_sentinel_pass_pinned_core(self):
        for response in ([0xff], [2, 0, 1], [0, 1, 2]):
            request, accepted, result = native._protocol_sort_card_core_fixture(
                3, response)
            self.assertEqual(request[:3], [25, 0, 3])
            self.assertTrue(accepted)
            self.assertEqual(result, [])
        _, accepted, result = native._protocol_sort_card_core_fixture(
            3, [0, 0, 2])
        self.assertFalse(accepted)
        self.assertEqual(result, [1])

    def test_sort_frame_through_actual_adapter_and_same_core_instance(self):
        frame, branch_sizes, accepted = native._protocol_sort_adapter_fixture(
            3, [2, 0, 0])
        self.assertEqual(frame[:3], [25, 0, 3])
        self.assertEqual(branch_sizes, [4, 2, 1])
        self.assertTrue(accepted)
        frame, branch_sizes, accepted = native._protocol_sort_adapter_fixture(
            3, [3])
        self.assertEqual(branch_sizes, [4])
        self.assertTrue(accepted)

    def test_multi_place_mapping_covers_both_players_and_zone_types(self):
        # One open bit in each protocol byte: own M/S then opponent M/S.
        disabled = 0xFFFFFFFF
        for bit in (0, 8, 16, 24):
            disabled &= ~(1 << bit)
        places = native._protocol_usable_places(disabled, False)
        self.assertEqual(places, [1, 8, 16, 23])
        self.assertEqual(native._protocol_encode_place(1, 0), [0, 4, 0])
        self.assertEqual(native._protocol_encode_place(8, 0), [0, 8, 0])
        self.assertEqual(native._protocol_encode_place(16, 0), [1, 4, 0])
        self.assertEqual(native._protocol_encode_place(23, 0), [1, 8, 0])
        self.assertEqual(native._protocol_encode_place(16, 1), [0, 4, 0])

    def test_place_and_disfield_frames_through_adapter_and_core(self):
        disabled = 0xFFFFFFFF
        for bit in (0, 8, 16, 24):
            disabled &= ~(1 << bit)
        for disfield, message in ((False, 18), (True, 24)):
            frame, sizes, accepted = native._protocol_place_adapter_fixture(
                disfield, disabled, 0, [4])
            self.assertEqual(frame[:3], [message, 0, 0])
            self.assertEqual(sizes, [5])
            self.assertTrue(accepted)

            frame, sizes, accepted = native._protocol_place_adapter_fixture(
                disfield, disabled, 1, [3])
            self.assertEqual(frame[:3], [message, 0, 1])
            self.assertEqual(sizes, [4])
            self.assertTrue(accepted)

            frame, sizes, accepted = native._protocol_place_adapter_fixture(
                disfield, disabled, 2, [0, 2])
            self.assertEqual(frame[:3], [message, 0, 2])
            self.assertEqual(sizes, [4, 3])
            self.assertTrue(accepted)

            pendulum_disabled = 0xFFFFFFFF
            for bit in (6, 14, 22, 30):
                pendulum_disabled &= ~(1 << bit)
            for choice in range(4):
                _, sizes, accepted = native._protocol_place_adapter_fixture(
                    disfield, pendulum_disabled, 1, [choice])
                self.assertEqual(sizes, [4])
                self.assertTrue(accepted)

    def test_composite_position_mask_yields_only_atomic_positions(self):
        self.assertEqual(native._protocol_position_options(0xF), [1, 2, 4, 8])
        self.assertEqual(native._protocol_position_options(0x6), [2, 4])

    def test_core_automatic_short_circuits_emit_no_adapter_frame(self):
        self.assertEqual(
            list(native._protocol_automatic_selection_fixture()),
            [True, True, True])

    def test_race_and_attribute_masks_preserve_exact_popcount(self):
        race_options = native._protocol_exact_mask_options(
            (1 << 0) | (1 << 7) | (1 << 31), 2, 26)
        self.assertEqual(race_options, [(1 << 0) | (1 << 7)])
        self.assertEqual(
            native._protocol_exact_mask_options(1 << 31, 0, 26), [0])
        self.assertEqual(native._protocol_exact_mask_options(0, 0, 26), [0])
        self.assertEqual(
            native._protocol_exact_mask_options(0b1000101, 2, 7),
            [0b0000101, 0b1000001, 0b1000100],
        )

    def test_race_core_ignores_undefined_high_bits_in_count_and_membership(self):
        high = 1 << 31
        request, accepted, result = native._protocol_announce_race_core_fixture(
            high | 1, 1, high | 1)
        self.assertEqual(request[:3], [140, 0, 1])
        self.assertTrue(accepted)
        self.assertEqual(int.from_bytes(bytes(result[-4:]), "little"), high | 1)
        self.assertEqual(
            native._protocol_exact_mask_options(high | 1, 1, 26), [1])

        _, accepted, result = native._protocol_announce_race_core_fixture(
            high | 1, 1, high)
        self.assertFalse(accepted)
        self.assertEqual(result, [1])
        _, accepted, result = native._protocol_announce_race_core_fixture(
            1, 1, 2)
        self.assertFalse(accepted)
        self.assertEqual(result, [1])

        request, accepted, _ = native._protocol_announce_race_core_fixture(
            high, 1, high)
        self.assertEqual(request[2], 0)
        self.assertTrue(accepted)
        self.assertEqual(
            native._protocol_exact_mask_options(high, 0, 26), [0])

    def test_mask_frames_through_actual_adapter_and_same_core(self):
        race_request, race_actions, race_response, race_accepted = (
            native._protocol_mask_adapter_fixture(
                True, (1 << 31) | 0b101, 1, 1))
        self.assertEqual(race_request[:3], [140, 0, 1])
        self.assertEqual(race_actions, 2)
        self.assertEqual(race_response, 0b100)
        self.assertTrue(race_accepted)
        attr_request, attr_actions, attr_response, attr_accepted = (
            native._protocol_mask_adapter_fixture(False, 0b10101, 2, 1))
        self.assertEqual(attr_request[:3], [141, 0, 2])
        self.assertEqual(attr_actions, 3)
        self.assertEqual(attr_response, 0b10001)
        self.assertTrue(attr_accepted)

    def test_large_race_choice_count_fails_before_materialization(self):
        self.assertEqual(
            native._protocol_bounded_combination_count(26, 13, 24), 25)
        self.assertEqual(
            native._protocol_bounded_combination_count(4, 2, 24), 6)
        with self.assertRaisesRegex(RuntimeError, "25 legal actions exceed model capacity 24"):
            native._protocol_validate_action_capacity(
                native._protocol_bounded_combination_count(26, 13, 24), 24)

    def test_sort_permutations_and_no_sort_sentinel(self):
        responses = native._protocol_sort_responses(3)
        self.assertEqual(len(responses), 7)
        self.assertEqual(responses[-1], [0xFF])
        self.assertEqual(
            {tuple(value) for value in responses[:-1]},
            set(itertools.permutations(range(3))),
        )

    def test_announce_card_rpn_filters(self):
        opcode_and = 0x40000004
        opcode_not = 0x40000007
        opcode_iscode = 0x40000100
        opcode_issetcard = 0x40000101
        opcode_istype = 0x40000102
        opcode_israce = 0x40000103
        opcode_isattribute = 0x40000104
        monster = 0x1
        token = 0x4000
        code = 12345678
        common = (code, 0, 0x123, monster, 1 << 24, 0x20)
        self.assertTrue(native._protocol_is_declarable(
            *common, [code, opcode_iscode]))
        self.assertTrue(native._protocol_is_declarable(
            *common, [0x123, opcode_issetcard]))
        self.assertTrue(native._protocol_is_declarable(
            *common, [monster, opcode_istype]))
        self.assertTrue(native._protocol_is_declarable(
            *common, [1 << 24, opcode_israce]))
        self.assertTrue(native._protocol_is_declarable(
            *common, [0x20, opcode_isattribute]))
        self.assertTrue(native._protocol_is_declarable(
            *common,
            [monster, opcode_istype, token, opcode_istype,
             opcode_not, opcode_and],
        ))
        self.assertFalse(native._protocol_is_declarable(
            code, code + 100, 0x123, monster, 1 << 24, 0x20,
            [monster, opcode_istype],
        ))
        self.assertFalse(native._protocol_is_declarable(
            code, 0, 0x123, monster | token, 1 << 24, 0x20,
            [monster, opcode_istype],
        ))


if __name__ == "__main__":
    unittest.main()
