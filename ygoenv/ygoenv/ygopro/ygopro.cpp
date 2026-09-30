#include "ygoenv/ygopro/ygopro.h"
#include "ygoenv/core/py_envpool.h"
#include <pybind11/stl.h>

using YGOProEnvSpec = PyEnvSpec<ygopro::YGOProEnvSpec>;
using YGOProEnvPool = PyEnvPool<ygopro::YGOProEnvPool>;

PYBIND11_MODULE(ygopro_ygoenv, m) {
  REGISTER(m, YGOProEnvSpec, YGOProEnvPool)

  m.def("init_module", &ygopro::init_module);
  m.def("_protocol_sum_check", &ygopro::core_select_sum_check);
  m.def("_protocol_sum_limit_check", &ygopro::core_select_sum_limit_check);
  m.def("_protocol_sum_sequences", &ygopro::core_select_sum_sequences);
  m.def("_protocol_sum_limit_combinations",
        &ygopro::core_select_sum_limit_combinations);
  m.def("_protocol_tribute_combinations",
        &ygopro::core_select_tribute_combinations);
  m.def("_protocol_weighted_staged_choices",
        [](int kind, const std::vector<uint32_t> &must,
           const std::vector<uint32_t> &optional,
           const std::vector<int> &selected, int32_t target,
           int min_count, int max_count, size_t visit_limit) {
          if (kind < 0 || kind > 2) {
            throw std::runtime_error("Invalid weighted selection kind");
          }
          return ygopro::core_weighted_staged_choices(
              static_cast<ygopro::CoreWeightedKind>(kind), must, optional,
              selected, target, min_count, max_count, visit_limit);
        });
  m.def("_protocol_weighted_core_fixture",
        [](int kind, const std::vector<uint32_t> &must,
           const std::vector<uint32_t> &optional, int32_t target,
           int min_count, int max_count, const std::vector<int> &selected) {
          if (kind < 0 || kind > 2) {
            throw std::runtime_error("Invalid weighted selection kind");
          }
          return ygopro::core_weighted_core_fixture(
              static_cast<ygopro::CoreWeightedKind>(kind), must, optional,
              target, min_count, max_count, selected);
        });
  m.def("_protocol_counter_allocation",
        &ygopro::core_select_counter_allocation);
  m.def("_protocol_position_options", &ygopro::core_position_options);
  m.def("_protocol_automatic_selection_fixture",
        &ygopro::core_automatic_selection_fixture);
  m.def("_protocol_exact_mask_options", &ygopro::core_exact_mask_options);
  m.def("_protocol_announce_race_core_fixture",
        &ygopro::core_announce_race_core_fixture);
  m.def("_protocol_mask_adapter_fixture",
        &ygopro::core_mask_adapter_fixture);
  m.def("_protocol_number_adapter_fixture",
        &ygopro::core_number_adapter_fixture);
  m.def("_protocol_position_adapter_fixture",
        &ygopro::core_position_adapter_fixture);
  m.def("_protocol_place_adapter_fixture",
        &ygopro::core_place_adapter_fixture);
  m.def("_protocol_scalar_adapter_fixture",
        &ygopro::core_scalar_adapter_fixture);
  m.def("_protocol_command_adapter_fixture",
        &ygopro::core_command_adapter_fixture);
  m.def("_protocol_chain_adapter_fixture",
        &ygopro::core_chain_adapter_fixture);
  m.def("_protocol_unknown_card_placeholder", []() {
    return std::make_pair(ygopro::c_get_card(0).type(),
                          ygopro::c_get_card_id(0));
  });
  m.def("_protocol_announce_card_adapter_fixture",
        &ygopro::core_announce_card_adapter_fixture);
  m.def("_protocol_announce_card_large_adapter_fixture",
        &ygopro::core_announce_card_large_adapter_fixture);
  m.def("_protocol_bounded_combination_count",
        &ygopro::core_bounded_combination_count);
  m.def("_protocol_sort_responses", &ygopro::core_sort_responses);
  m.def("_protocol_sort_card_core_fixture",
        &ygopro::core_sort_card_core_fixture);
  m.def("_protocol_sort_adapter_fixture",
        &ygopro::core_sort_adapter_fixture);
  m.def("_protocol_counter_adapter_fixture",
        &ygopro::core_counter_adapter_fixture);
  m.def("_protocol_weighted_adapter_fixture",
        [](int kind, const std::vector<uint32_t> &must,
           const std::vector<uint32_t> &optional, int32_t target,
           int min_count, int max_count, const std::vector<int> &choices) {
          if (kind < 0 || kind > 2) {
            throw std::runtime_error("Invalid weighted selection kind");
          }
          return ygopro::core_weighted_adapter_fixture(
              static_cast<ygopro::CoreWeightedKind>(kind), must, optional,
              target, min_count, max_count, choices);
        });
  m.def("_protocol_tribute_adapter_fixture",
        [](const std::vector<uint32_t> &optional, int target,
           int max_count, bool cancelable,
           const std::vector<int> &choices) {
          return ygopro::core_weighted_adapter_fixture(
              ygopro::CoreWeightedKind::Tribute, {}, optional, target,
              0, max_count, choices, cancelable);
        });
  m.def("_protocol_select_counter_core_fixture",
        &ygopro::core_select_counter_core_fixture);
  m.def("_protocol_index_response", &ygopro::core_index_response);
  m.def("_protocol_command_response", &ygopro::core_command_response);
  m.def("_protocol_uint32_identity", &ygopro::core_uint32_identity);
  m.def("_protocol_scalar_response", &ygopro::core_scalar_response);
  m.def("_protocol_message_id", &ygopro::msg_to_id);
  m.def("_protocol_validate_action_capacity",
        &ygopro::core_validate_action_capacity);
  m.def("_protocol_select_unselect_response_index",
        &ygopro::core_select_unselect_response_index);
  m.def("_protocol_select_unselect_core_fixture",
        &ygopro::core_select_unselect_core_fixture);
  m.def("_protocol_select_unselect_adapter_fixture",
        &ygopro::core_select_unselect_adapter_fixture);
  m.def("_protocol_select_card_adapter_fixture",
        &ygopro::core_select_card_adapter_fixture);
  m.def("_protocol_rock_paper_scissors_options",
        &ygopro::core_rock_paper_scissors_options);
  m.def("_protocol_rock_paper_scissors_player",
        &ygopro::core_rock_paper_scissors_player);
  m.def("_protocol_rock_paper_scissors_core_fixture",
        &ygopro::core_rock_paper_scissors_fixture);
  m.def("_protocol_rps_adapter_fixture",
        &ygopro::core_rps_adapter_fixture);
  m.def("_protocol_notification_adapter_fixture",
        &ygopro::core_notification_adapter_fixture);
  m.def("_protocol_cancel_history_encoding_fixture",
        &ygopro::core_cancel_history_encoding_fixture);
  m.def("_protocol_policy_cancel_filter_fixture",
        &ygopro::core_policy_cancel_filter_fixture);
  m.def("_protocol_usable_places", [](uint32_t flag, bool reverse) {
    std::vector<int> values;
    for (auto place : ygopro::flag_to_usable_places(flag, reverse)) {
      values.push_back(static_cast<int>(place));
    }
    return values;
  });
  m.def("_protocol_encode_place", [](int place, uint8_t player) {
    return ygopro::core_encode_place(
        static_cast<ygopro::ActionPlace>(place), player);
  });
  m.def("_protocol_is_declarable",
        [](uint32_t code, uint32_t alias, uint64_t setcode, uint32_t type,
           uint32_t race, uint32_t attribute,
           const std::vector<uint32_t> &opcodes) {
          card_data card;
          card.code = code;
          card.alias = alias;
          card.set_setcode(setcode);
          card.type = type;
          card.race = race;
          card.attribute = attribute;
          return ygopro::is_declarable(card, opcodes);
        });
  m.def("_greedy_action_fixture", []() {
    ygopro::GreedyAI ai("fixture", 8000, 0);
    std::vector<ygopro::LegalAction> idle = {
        ygopro::LegalAction::act_spec(ygopro::ActionAct::Summon, "h1"),
        ygopro::LegalAction::phase(ygopro::ActionPhase::Battle),
        ygopro::LegalAction::phase(ygopro::ActionPhase::End)};
    std::vector<ygopro::LegalAction> battle = {
        ygopro::LegalAction::activate_spec(0, "m1"),
        ygopro::LegalAction::act_spec(ygopro::ActionAct::Attack, "m2"),
        ygopro::LegalAction::phase(ygopro::ActionPhase::End)};
    std::vector<ygopro::LegalAction> optional = {
        ygopro::LegalAction::cancel(),
        ygopro::LegalAction::from_spec("m1")};
    return std::vector<int>{ai.think(idle), ai.think(battle),
                            ai.think(optional)};
  });
}
