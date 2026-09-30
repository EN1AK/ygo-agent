#ifndef YGOENV_YGOPRO_YGOPRO_H_
#define YGOENV_YGOPRO_YGOPRO_H_

// clang-format off
#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <numeric>
#include <stdexcept>
#include <string>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <set>
#include <stack>
#include <sstream>
#include <vector>

#ifndef _WIN32
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif


#include <fmt/core.h>
#include <fmt/ranges.h>
#include <SQLiteCpp/SQLiteCpp.h>
#include <SQLiteCpp/VariadicBind.h>
#include <ankerl/unordered_dense.h>
#include <unordered_set>
#include <utility>

#include "ygoenv/core/BS_thread_pool.h"

#include "ygoenv/core/async_envpool.h"
#include "ygoenv/core/env.h"

#include "ygopro-core/common.h"
#include "ygopro-core/card_data.h"
#include "ygopro-core/card.h"
#include "ygopro-core/duel.h"
#include "ygopro-core/effect.h"
#include "ygopro-core/field.h"
#include "ygopro-core/ocgapi.h"

// clang-format on

namespace ygopro {

inline std::vector<std::vector<int>> combinations(int n, int r) {
  std::vector<std::vector<int>> combs;
  std::vector<bool> m(n);
  std::fill(m.begin(), m.begin() + r, true);

  do {
    std::vector<int> cs;
    cs.reserve(r);
    for (int i = 0; i < n; ++i) {
      if (m[i]) {
        cs.push_back(i);
      }
    }
    combs.push_back(cs);
  } while (std::prev_permutation(m.begin(), m.end()));

  return combs;
}

inline bool sum_to(const std::vector<int> &w, const std::vector<int> ind, int i,
                   int r) {
  if (r <= 0) {
    return false;
  }
  int n = ind.size();
  if (i == n - 1) {
    return r == 1 || (w[ind[i]] == r);
  }
  return sum_to(w, ind, i + 1, r - 1) || sum_to(w, ind, i + 1, r - w[ind[i]]);
}

inline bool sum_to(const std::vector<int> &w, const std::vector<int> ind,
                   int r) {
  return sum_to(w, ind, 0, r);
}

inline std::vector<std::vector<int>>
combinations_with_weight(const std::vector<int> &weights, int r) {
  int n = weights.size();
  std::vector<std::vector<int>> results;

  for (int k = 1; k <= n; k++) {
    std::vector<std::vector<int>> combs = combinations(n, k);
    for (const auto &comb : combs) {
      if (sum_to(weights, comb, r)) {
        results.push_back(comb);
      }
    }
  }
  return results;
}

inline bool sum_to2(const std::vector<std::vector<int>> &w,
                    const std::vector<int> ind, int i, int r) {
  if (r <= 0) {
    return false;
  }
  int n = ind.size();
  const auto &w_ = w[ind[i]];
  if (i == n - 1) {
    if (w_.size() == 1) {
      return w_[0] == r;
    } else {
      return w_[0] == r || w_[1] == r;
    }
  }
  if (w_.size() == 1) {
    return sum_to2(w, ind, i + 1, r - w_[0]);
  } else {
    return sum_to2(w, ind, i + 1, r - w_[0]) ||
           sum_to2(w, ind, i + 1, r - w_[1]);
  }
}

inline bool sum_to2(const std::vector<std::vector<int>> &w,
                    const std::vector<int> ind, int r) {
  return sum_to2(w, ind, 0, r);
}

inline std::vector<std::vector<int>>
combinations_with_weight2(
  const std::vector<std::vector<int>> &weights, int r) {
  int n = weights.size();
  std::vector<std::vector<int>> results;

  for (int k = 1; k <= n; k++) {
    std::vector<std::vector<int>> combs = combinations(n, k);
    for (const auto &comb : combs) {
      if (sum_to2(weights, comb, r)) {
        results.push_back(comb);
      }
    }
  }
  return results;
}

inline bool core_select_sum_check(const std::vector<uint32_t> &params,
                                  size_t index, int32_t acc,
                                  int32_t opmin = 0xffff) {
  if (acc == 0 || index >= params.size()) return false;
  const int32_t o1 = params[index] & 0xffff;
  const int32_t o2 = params[index] >> 16;
  if (index == params.size() - 1) {
    return (acc == o1 && acc + opmin > o1) ||
           (o2 && acc == o2 && acc + opmin > o2);
  }
  return (acc > o1 && core_select_sum_check(
                          params, index + 1, acc - o1, std::min(o1, opmin))) ||
         (o2 > 0 && acc > o2 && core_select_sum_check(
                              params, index + 1, acc - o2,
                              std::min(o2, opmin)));
}

inline bool core_select_sum_limit_check(const std::vector<uint32_t> &params,
                                        int32_t acc) {
  if (params.empty()) return false;
  int32_t sum = 0;
  int32_t maximum = 0;
  int32_t minimum = 0x7fffffff;
  for (uint32_t param : params) {
    const int32_t o1 = param & 0xffff;
    const int32_t o2 = param >> 16;
    const int32_t smaller = (o2 && o2 < o1) ? o2 : o1;
    sum += smaller;
    maximum += std::max(o1, o2);
    minimum = std::min(minimum, smaller);
  }
  return maximum >= acc && sum - minimum < acc;
}

inline std::vector<std::vector<int>> core_select_sum_sequences(
    const std::vector<uint32_t> &must_params,
    const std::vector<uint32_t> &optional_params, int32_t acc,
    int min_count, int max_count) {
  std::vector<std::vector<int>> results;
  std::vector<int> current;
  max_count = std::min<int>(max_count, optional_params.size());
  min_count = std::max(0, min_count);

  auto append_if_valid = [&]() {
    std::vector<uint32_t> params = must_params;
    for (int index : current) params.push_back(optional_params[index]);
    if (core_select_sum_check(params, 0, acc)) {
      results.push_back(current);
    }
  };

  // Generate each selected card set once.  The old implementation generated
  // every permutation (n! in the common max_count == n case), even though card
  // selection order has no gameplay meaning.  Large synchro-material prompts
  // could therefore spin a worker indefinitely in MSG_SELECT_SUM.
  auto visit = [&](auto &&self, int next) -> void {
    if (static_cast<int>(current.size()) >= min_count) {
      append_if_valid();
    }
    if (static_cast<int>(current.size()) == max_count) return;
    for (int index = next; index < static_cast<int>(optional_params.size());
         ++index) {
      current.push_back(index);
      self(self, index + 1);
      current.pop_back();
    }
  };
  if (min_count <= max_count) visit(visit, 0);
  return results;
}

inline std::vector<std::vector<int>> core_select_sum_limit_combinations(
    const std::vector<uint32_t> &must_params,
    const std::vector<uint32_t> &optional_params, int32_t acc) {
  std::vector<std::vector<int>> results;
  for (int count = 0; count <= static_cast<int>(optional_params.size()); ++count) {
    for (const auto &combination : combinations(optional_params.size(), count)) {
      std::vector<uint32_t> params = must_params;
      for (int index : combination) params.push_back(optional_params[index]);
      if (core_select_sum_limit_check(params, acc)) results.push_back(combination);
    }
  }
  return results;
}

inline std::vector<std::vector<int>> core_select_tribute_combinations(
    const std::vector<int> &release_params, int min_value, int max_cards) {
  std::vector<std::vector<int>> results;
  max_cards = std::min<int>(max_cards, release_params.size());
  for (int selected = 0; selected <= max_cards; ++selected) {
    for (const auto &combination : combinations(release_params.size(), selected)) {
      int total = 0;
      for (int index : combination) total += release_params[index];
      if (total >= min_value) results.push_back(combination);
    }
  }
  return results;
}

// The production path must never materialize the power set of selectable
// cards. A staged choice exposes only indices that admit a core-valid suffix.
// The search has a hard work budget: exhaustion is a protocol boundary, not a
// silent loss of legal actions.
enum class CoreWeightedKind { Tribute, ExactSum, SumLimit };

inline bool core_weighted_valid(CoreWeightedKind kind,
                                const std::vector<uint32_t> &must,
                                const std::vector<uint32_t> &optional,
                                const std::vector<int> &selected,
                                int32_t target, int min_count, int max_count) {
  if (selected.size() + must.size() > 255) return false;
  if (kind == CoreWeightedKind::Tribute) {
    if (selected.size() > static_cast<size_t>(max_count)) return false;
    int total = 0;
    for (int index : selected) total += optional[index];
    return total >= target;
  }
  std::vector<uint32_t> params = must;
  for (int index : selected) params.push_back(optional[index]);
  if (kind == CoreWeightedKind::ExactSum) {
    return selected.size() >= static_cast<size_t>(min_count) &&
           selected.size() <= static_cast<size_t>(max_count) &&
           core_select_sum_check(params, 0, target);
  }
  return core_select_sum_limit_check(params, target);
}

inline bool core_weighted_has_completion(
    CoreWeightedKind kind, const std::vector<uint32_t> &must,
    const std::vector<uint32_t> &optional, std::vector<int> &selected,
    int32_t target, int min_count, int max_count, size_t &visits,
    size_t visit_limit) {
  if (++visits > visit_limit) {
    throw std::runtime_error("[protocol boundary] weighted selection search budget exhausted");
  }
  if (core_weighted_valid(kind, must, optional, selected, target,
                          min_count, max_count)) return true;
  if (kind != CoreWeightedKind::SumLimit &&
      selected.size() >= static_cast<size_t>(max_count)) return false;
  if (selected.size() + must.size() >= 255) return false;
  const int next = selected.empty() ? 0 : selected.back() + 1;
  if (kind == CoreWeightedKind::ExactSum &&
      selected.size() + optional.size() - next <
          static_cast<size_t>(min_count)) return false;
  for (int index = next; index < static_cast<int>(optional.size()); ++index) {
    selected.push_back(index);
    const bool possible = core_weighted_has_completion(
        kind, must, optional, selected, target, min_count, max_count,
        visits, visit_limit);
    selected.pop_back();
    if (possible) return true;
  }
  return false;
}

inline std::pair<std::vector<int>, bool> core_weighted_staged_choices(
    CoreWeightedKind kind, const std::vector<uint32_t> &must,
    const std::vector<uint32_t> &optional, const std::vector<int> &selected,
    int32_t target, int min_count, int max_count,
    size_t visit_limit = 1000000) {
  const bool finishable = core_weighted_valid(
      kind, must, optional, selected, target, min_count, max_count);
  std::vector<int> choices;
  size_t visits = 0;
  const int next = selected.empty() ? 0 : selected.back() + 1;
  for (int index = next; index < static_cast<int>(optional.size()); ++index) {
    auto candidate = selected;
    candidate.push_back(index);
    if (core_weighted_has_completion(kind, must, optional, candidate, target,
                                     min_count, max_count, visits,
                                     visit_limit)) choices.push_back(index);
  }
  return {choices, finishable};
}

inline std::vector<uint16_t> core_select_counter_allocation(
    const std::vector<int> &capacities, int requested) {
  std::vector<uint16_t> allocation;
  allocation.reserve(capacities.size());
  int remaining = requested;
  for (int capacity : capacities) {
    const uint16_t value = static_cast<uint16_t>(
        std::min(remaining, std::max(0, capacity)));
    allocation.push_back(value);
    remaining -= value;
  }
  if (remaining != 0) {
    throw std::runtime_error(fmt::format(
        "Counter allocation requested {} but candidates only hold {}",
        requested, requested - remaining));
  }
  return allocation;
}

inline bool is_declarable(const card_data &card,
                          const std::vector<uint32_t> &opcodes) {
  // RPN evaluator matching ygopro-core's announce-card validator.  Supporting
  // the full expression language is necessary for filters such as "not an
  // Extra Deck monster", not only explicit chains of OPCODE_ISCODE.
  std::stack<int32_t> values;
  for (uint32_t opcode : opcodes) {
    auto binary = [&values](auto operation) {
      if (values.size() < 2) return;
      int32_t rhs = values.top(); values.pop();
      int32_t lhs = values.top(); values.pop();
      values.push(operation(lhs, rhs));
    };
    switch (opcode) {
    case OPCODE_ADD: binary([](int32_t a, int32_t b) { return a + b; }); break;
    case OPCODE_SUB: binary([](int32_t a, int32_t b) { return a - b; }); break;
    case OPCODE_MUL: binary([](int32_t a, int32_t b) { return a * b; }); break;
    case OPCODE_DIV:
      binary([](int32_t a, int32_t b) { return b == 0 ? 0 : a / b; }); break;
    case OPCODE_AND: binary([](int32_t a, int32_t b) { return a && b; }); break;
    case OPCODE_OR: binary([](int32_t a, int32_t b) { return a || b; }); break;
    case OPCODE_NEG:
      if (!values.empty()) { int32_t v = values.top(); values.pop(); values.push(-v); }
      break;
    case OPCODE_NOT:
      if (!values.empty()) { int32_t v = values.top(); values.pop(); values.push(!v); }
      break;
    case OPCODE_ISCODE:
      if (!values.empty()) { int32_t v = values.top(); values.pop(); values.push(card.code == static_cast<uint32_t>(v)); }
      break;
    case OPCODE_ISSETCARD:
      if (!values.empty()) { int32_t v = values.top(); values.pop(); values.push(card.is_setcode(v)); }
      break;
    case OPCODE_ISTYPE:
      if (!values.empty()) { int32_t v = values.top(); values.pop(); values.push(card.type & v); }
      break;
    case OPCODE_ISRACE:
      if (!values.empty()) { int32_t v = values.top(); values.pop(); values.push(card.race & v); }
      break;
    case OPCODE_ISATTRIBUTE:
      if (!values.empty()) { int32_t v = values.top(); values.pop(); values.push(card.attribute & v); }
      break;
    default: values.push(static_cast<int32_t>(opcode)); break;
    }
  }
  if (values.size() != 1 || values.top() == 0) return false;
  return card.code == 78734254u || card.code == 13857930u ||
         (!card.alias && (card.type & (TYPE_MONSTER | TYPE_TOKEN)) !=
                             (TYPE_MONSTER | TYPE_TOKEN));
}

static std::string msg_to_string(int msg) {
  switch (msg) {
  case MSG_RETRY:
    return "retry";
  case MSG_HINT:
    return "hint";
  case MSG_WIN:
    return "win";
  case MSG_SELECT_BATTLECMD:
    return "select_battlecmd";
  case MSG_SELECT_IDLECMD:
    return "select_idlecmd";
  case MSG_SELECT_EFFECTYN:
    return "select_effectyn";
  case MSG_SELECT_YESNO:
    return "select_yesno";
  case MSG_SELECT_OPTION:
    return "select_option";
  case MSG_SELECT_CARD:
    return "select_card";
  case MSG_SELECT_CHAIN:
    return "select_chain";
  case MSG_SELECT_PLACE:
    return "select_place";
  case MSG_SELECT_POSITION:
    return "select_position";
  case MSG_SELECT_TRIBUTE:
    return "select_tribute";
  case MSG_SELECT_COUNTER:
    return "select_counter";
  case MSG_SELECT_SUM:
    return "select_sum";
  case MSG_SELECT_DISFIELD:
    return "select_disfield";
  case MSG_SORT_CARD:
    return "sort_card";
  case MSG_SELECT_UNSELECT_CARD:
    return "select_unselect_card";
  case MSG_CONFIRM_DECKTOP:
    return "confirm_decktop";
  case MSG_CONFIRM_CARDS:
    return "confirm_cards";
  case MSG_SHUFFLE_DECK:
    return "shuffle_deck";
  case MSG_SHUFFLE_HAND:
    return "shuffle_hand";
  case MSG_SWAP_GRAVE_DECK:
    return "swap_grave_deck";
  case MSG_SHUFFLE_SET_CARD:
    return "shuffle_set_card";
  case MSG_REVERSE_DECK:
    return "reverse_deck";
  case MSG_DECK_TOP:
    return "deck_top";
  case MSG_SHUFFLE_EXTRA:
    return "shuffle_extra";
  case MSG_NEW_TURN:
    return "new_turn";
  case MSG_NEW_PHASE:
    return "new_phase";
  case MSG_CONFIRM_EXTRATOP:
    return "confirm_extratop";
  case MSG_MOVE:
    return "move";
  case MSG_POS_CHANGE:
    return "pos_change";
  case MSG_SET:
    return "set";
  case MSG_SWAP:
    return "swap";
  case MSG_FIELD_DISABLED:
    return "field_disabled";
  case MSG_SUMMONING:
    return "summoning";
  case MSG_SUMMONED:
    return "summoned";
  case MSG_SPSUMMONING:
    return "spsummoning";
  case MSG_SPSUMMONED:
    return "spsummoned";
  case MSG_FLIPSUMMONING:
    return "flipsummoning";
  case MSG_FLIPSUMMONED:
    return "flipsummoned";
  case MSG_CHAINING:
    return "chaining";
  case MSG_CHAINED:
    return "chained";
  case MSG_CHAIN_SOLVING:
    return "chain_solving";
  case MSG_CHAIN_SOLVED:
    return "chain_solved";
  case MSG_CHAIN_END:
    return "chain_end";
  case MSG_CHAIN_NEGATED:
    return "chain_negated";
  case MSG_CHAIN_DISABLED:
    return "chain_disabled";
  case MSG_RANDOM_SELECTED:
    return "random_selected";
  case MSG_BECOME_TARGET:
    return "become_target";
  case MSG_DRAW:
    return "draw";
  case MSG_DAMAGE:
    return "damage";
  case MSG_RECOVER:
    return "recover";
  case MSG_EQUIP:
    return "equip";
  case MSG_LPUPDATE:
    return "lpupdate";
  case MSG_CARD_TARGET:
    return "card_target";
  case MSG_CANCEL_TARGET:
    return "cancel_target";
  case MSG_PAY_LPCOST:
    return "pay_lpcost";
  case MSG_ADD_COUNTER:
    return "add_counter";
  case MSG_REMOVE_COUNTER:
    return "remove_counter";
  case MSG_ATTACK:
    return "attack";
  case MSG_BATTLE:
    return "battle";
  case MSG_ATTACK_DISABLED:
    return "attack_disabled";
  case MSG_DAMAGE_STEP_START:
    return "damage_step_start";
  case MSG_DAMAGE_STEP_END:
    return "damage_step_end";
  case MSG_MISSED_EFFECT:
    return "missed_effect";
  case MSG_TOSS_COIN:
    return "toss_coin";
  case MSG_TOSS_DICE:
    return "toss_dice";
  case MSG_ROCK_PAPER_SCISSORS:
    return "rock_paper_scissors";
  case MSG_HAND_RES:
    return "hand_res";
  case MSG_ANNOUNCE_RACE:
    return "announce_race";
  case MSG_ANNOUNCE_ATTRIB:
    return "announce_attrib";
  case MSG_ANNOUNCE_CARD:
    return "announce_card";
  case MSG_ANNOUNCE_NUMBER:
    return "announce_number";
  case MSG_CARD_HINT:
    return "card_hint";
  case MSG_TAG_SWAP:
    return "tag_swap";
  case MSG_RELOAD_FIELD:
    return "reload_field";
  case MSG_AI_NAME:
    return "ai_name";
  case MSG_SHOW_HINT:
    return "show_hint";
  case MSG_PLAYER_HINT:
    return "player_hint";
  case MSG_MATCH_KILL:
    return "match_kill";
  case MSG_CUSTOM_MSG:
    return "custom_msg";
  default:
    return "unknown_msg";
  }
}

// system string
static const std::map<int, std::string> system_strings = {
    // announce type
    {1050, "Monster"},
    {1051, "Spell"},
    {1052, "Trap"},
    {1054, "Normal"},
    {1055, "Effect"},
    {1056, "Fusion"},
    {1057, "Ritual"},
    {1058, "Trap Monsters"},
    {1059, "Spirit"},
    {1060, "Union"},
    {1061, "Gemini"},
    {1062, "Tuner"},
    {1063, "Synchro"},
    {1064, "Token"},
    {1066, "Quick-Play"},
    {1067, "Continuous"},
    {1068, "Equip"},
    {1069, "Field"},
    {1070, "Counter"},
    {1071, "Flip"},
    {1072, "Toon"},
    {1073, "Xyz"},
    {1074, "Pendulum"},
    {1075, "Special Summon"},
    {1076, "Link"},
    {1080, "(N/A)"},
    {1081, "Extra Monster Zone"},
    // announce type end
    // actions
    {1150, "Activate"},
    {1151, "Normal Summon"},
    {1152, "Special Summon"},
    {1153, "Set"},
    {1154, "Flip Summon"},
    {1155, "To Defense"},
    {1156, "To Attack"},
    {1157, "Attack"},
    {1158, "View"},
    {1159, "S/T Set"},
    {1160, "Put in Pendulum Zone"},
    {1161, "Do Effect"},
    {1162, "Reset Effect"},
    {1163, "Pendulum Summon"},
    {1164, "Synchro Summon"},
    {1165, "Xyz Summon"},
    {1166, "Link Summon"},
    {1167, "Tribute Summon"},
    {1168, "Ritual Summon"},
    {1169, "Fusion Summon"},
    {1190, "Add to hand"},
    {1191, "Send to GY"},
    {1192, "Banish"},
    {1193, "Return to Deck"},
    // actions end
    {1, "Normal Summon"},
    {30, "Replay rules apply. Continue this attack?"},
    {31, "Attack directly with this monster?"},
    {80, "Start Step of the Battle Phase."},
    {81, "During the End Phase."},
    {90, "Conduct this Normal Summon without Tributing?"},
    {91, "Use additional Summon?"},
    {92, "Tribute your opponent's monster?"},
    {93, "Continue selecting Materials?"},
    {94, "Activate this card's effect now?"},
    {95, "Use the effect of [%ls]?"},
    {96, "Use the effect of [%ls] to avoid destruction?"},
    {97, "Place [%ls] to a Spell & Trap Zone?"},
    {98, "Tribute a monster(s) your opponent controls?"},
    {200, "From [%ls], activate [%ls]?"},
    {203, "Chain another card or effect?"},
    {210, "Continue selecting?"},
    {218, "Pay LP by Effect of [%ls], instead?"},
    {219, "Detach Xyz material by Effect of [%ls], instead?"},
    {220, "Remove Counter(s) by Effect of [%ls], instead?"},
    {221, "On [%ls], Activate Trigger Effect of [%ls]?"},
    {222, "Activate Trigger Effect?"},
    {221, "On [%ls], Activate Trigger Effect of [%ls]?"},
    {1621, "Attack Negated"},
    {1622, "[%ls] Missed timing"}
};

static std::string get_system_string(int desc) {
  auto it = system_strings.find(desc);
  if (it != system_strings.end()) {
    return it->second;
  }
  return "system string " + std::to_string(desc);
}

static std::string ltrim(std::string s) {
  s.erase(s.begin(),
          std::find_if(s.begin(), s.end(), [](unsigned char ch) {
            return !std::isspace(ch);
          }));
  return s;
}


inline std::string ls_to_spec(uint8_t loc, uint8_t seq, uint8_t pos) {
  std::string spec;
  if (loc & LOCATION_HAND) {
    spec += "h";
  } else if (loc & LOCATION_MZONE) {
    spec += "m";
  } else if (loc & LOCATION_SZONE) {
    spec += "s";
  } else if (loc & LOCATION_GRAVE) {
    spec += "g";
  } else if (loc & LOCATION_REMOVED) {
    spec += "r";
  } else if (loc & LOCATION_EXTRA) {
    spec += "x";
  }
  spec += std::to_string(seq + 1);
  if (loc & LOCATION_OVERLAY) {
    spec.push_back('a' + pos);
  }
  return spec;
}

inline std::string ls_to_spec(uint8_t loc, uint8_t seq, uint8_t pos, bool opponent) {
  std::string spec = ls_to_spec(loc, seq, pos);
  if (opponent) {
    spec.insert(0, 1, 'o');
  }
  return spec;
}

inline std::tuple<uint8_t, uint8_t, uint8_t>
spec_to_ls(const std::string spec) {
  uint8_t loc;
  uint8_t seq;
  uint8_t pos = 0;
  int offset = 1;
  if (spec[0] == 'h') {
    loc = LOCATION_HAND;
  } else if (spec[0] == 'm') {
    loc = LOCATION_MZONE;
  } else if (spec[0] == 's') {
    loc = LOCATION_SZONE;
  } else if (spec[0] == 'g') {
    loc = LOCATION_GRAVE;
  } else if (spec[0] == 'r') {
    loc = LOCATION_REMOVED;
  } else if (spec[0] == 'x') {
    loc = LOCATION_EXTRA;
  } else if (std::isdigit(spec[0])) {
    loc = LOCATION_DECK;
    offset = 0;
  } else {
    std::string s = fmt::format("Invalid spec {}", spec);
    throw std::runtime_error(s);
  }
  int end = offset;
  while (end < spec.size() && std::isdigit(spec[end])) {
    end++;
  }
  seq = std::stoi(spec.substr(offset, end - offset)) - 1;
  if (end < spec.size()) {
    pos = spec[end] - 'a';
  }
  return {loc, seq, pos};
}


inline std::tuple<uint8_t, uint8_t, uint8_t, uint8_t>
spec_to_ls(uint8_t player, const std::string spec) {
  uint8_t controller = player;
  int offset = 0;
  if (spec[0] == 'o') {
    controller = 1 - player;
    offset++;
  }
  auto [loc, seq, pos] = spec_to_ls(spec.substr(offset));
  return {controller, loc, seq, pos};
}


static std::tuple<std::vector<uint32>, std::vector<uint32>, std::vector<uint32>> read_decks(const std::string &fp) {
  std::ifstream file(fp);
  std::string line;
  std::vector<uint32> main_deck, extra_deck, side_deck;
  bool found_extra = false;

  if (file.is_open()) {
    // Read the main deck
    while (std::getline(file, line)) {
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      if (line.find("side") != std::string::npos) {
        break;
      }
      if (line.find("extra") != std::string::npos) {
        found_extra = true;
        break;
      }
      // Check if line contains only digits
      if (std::all_of(line.begin(), line.end(), ::isdigit)) {
        main_deck.push_back(std::stoul(line));
      }
    }

    if (main_deck.size() < 40) {
      std::string err = fmt::format("Main deck must contain at least 40 cards, found: {}, file: {}", main_deck.size(), fp);
      throw std::runtime_error(err);
    }

    // Read the extra deck
    if (found_extra) {
      while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
          line.pop_back();
        }
        if (line.find("side") != std::string::npos) {
          break;
        }
        // Check if line contains only digits
        if (std::all_of(line.begin(), line.end(), ::isdigit)) {
          extra_deck.push_back(std::stoul(line));
        }
      }
    }

    // Read the side deck
    while (std::getline(file, line)) {
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      // Check if line contains only digits
      if (std::all_of(line.begin(), line.end(), ::isdigit)) {
        side_deck.push_back(std::stoul(line));
      }
    }

    file.close();
  } else {
    throw std::runtime_error(fmt::format("Unable to open deck file: {}", fp));
  }

  return std::make_tuple(main_deck, extra_deck, side_deck);
}

template <class K = uint8_t>
ankerl::unordered_dense::map<K, uint8_t>
make_ids(const std::map<K, std::string> &m, int id_offset = 0,
         int m_offset = 0) {
  ankerl::unordered_dense::map<K, uint8_t> m2;
  int i = 0;
  for (const auto &[k, v] : m) {
    if (i < m_offset) {
      i++;
      continue;
    }
    m2[k] = i - m_offset + id_offset;
    i++;
  }
  return m2;
}

template <class K = char>
ankerl::unordered_dense::map<K, uint8_t>
make_ids(const std::vector<K> &cmds, int id_offset = 0, int m_offset = 0) {
  ankerl::unordered_dense::map<K, uint8_t> m2;
  for (int i = m_offset; i < cmds.size(); i++) {
    m2[cmds[i]] = i - m_offset + id_offset;
  }
  return m2;
}

static std::string reason_to_string(uint8_t reason) {
  // !victory 0x0 Surrendered
  // !victory 0x1 LP reached 0
  // !victory 0x2 Cards can't be drawn
  // !victory 0x3 Time limit up
  // !victory 0x4 Lost connection
  switch (reason) {
  case 0x0:
    return "Surrendered";
  case 0x1:
    return "LP reached 0";
  case 0x2:
    return "Cards can't be drawn";
  case 0x3:
    return "Time limit up";
  case 0x4:
    return "Lost connection";
  default:
    return "Unknown";
  }
}

#define DEFINE_X_TO_ID_FUN(name, x_map) \
inline uint8_t name(decltype(x_map)::key_type x) { \
  auto it = x_map.find(x); \
  if (it != x_map.end()) { \
    return it->second; \
  } \
  throw std::runtime_error( \
    fmt::format("[" #name "] cannot find id: {}", x)); \
}

#define DEFINE_X_TO_STRING_FUN(name, x_map) \
inline std::string name(decltype(x_map)::key_type x) { \
  auto it = x_map.find(x); \
  if (it != x_map.end()) { \
    return it->second; \
  } \
  return "unknown"; \
}

static const ankerl::unordered_dense::map<int, uint8_t> system_string2id =
    make_ids(system_strings, 16);
inline uint8_t system_string_to_id(int desc) {
  auto it = system_string2id.find(desc);
  if (it != system_string2id.end()) return it->second;
  // Keep every historical hand-assigned ID stable.  The protocol permits
  // many more system descriptions than fit in a byte, so previously unseen
  // descriptions use a deterministic reserved hash bucket instead of
  // terminating a duel.  Card-specific descriptions retain their separate
  // 2..15 encoding path and do not pass through here.
  uint32_t mixed = static_cast<uint32_t>(desc) * 2654435761u;
  return static_cast<uint8_t>(128u + (mixed >> 25));
}


static const std::map<uint8_t, std::string> location2str = {
    {LOCATION_DECK, "Deck"},
    {LOCATION_HAND, "Hand"},
    {LOCATION_MZONE, "Main Monster Zone"},
    {LOCATION_SZONE, "Spell & Trap Zone"},
    {LOCATION_GRAVE, "Graveyard"},
    {LOCATION_REMOVED, "Banished"},
    {LOCATION_EXTRA, "Extra Deck"},
};

static const ankerl::unordered_dense::map<uint8_t, uint8_t> location2id =
    make_ids(location2str, 1);
DEFINE_X_TO_ID_FUN(location_to_id, location2id)


#define POS_NONE 0x0 // xyz materials (overlay) ???

static const std::map<uint8_t, std::string> position2str = {
    {POS_NONE, "none"},
    {POS_FACEUP_ATTACK, "face-up attack"},
    {POS_FACEDOWN_ATTACK, "face-down attack"},
    {POS_ATTACK, "attack"},
    {POS_FACEUP_DEFENSE, "face-up defense"},
    {POS_FACEUP, "face-up"},
    {POS_FACEDOWN_DEFENSE, "face-down defense"},
    {POS_FACEDOWN, "face-down"},
    {POS_DEFENSE, "defense"},
};
DEFINE_X_TO_STRING_FUN(position_to_string, position2str)

static const ankerl::unordered_dense::map<uint8_t, uint8_t> position2id =
    make_ids(position2str);
inline uint8_t position_to_id(uint8_t position) {
  auto it = position2id.find(position);
  if (it != position2id.end()) return it->second;
  // Some effects expose a legal-position mask rather than one resolved card
  // position (for example 0x06).  Preserve all historical IDs and reserve a
  // disjoint stable range for the remaining four-bit masks.
  if ((position & 0xf0) == 0) return static_cast<uint8_t>(9 + position);
  throw std::runtime_error(
      fmt::format("[position_to_id] invalid position mask: {}", position));
}


#define ATTRIBUTE_NONE 0x0 // token

static const std::map<uint8_t, std::string> attribute2str = {
    {ATTRIBUTE_NONE, "None"},   {ATTRIBUTE_EARTH, "Earth"},
    {ATTRIBUTE_WATER, "Water"}, {ATTRIBUTE_FIRE, "Fire"},
    {ATTRIBUTE_WIND, "Wind"},   {ATTRIBUTE_LIGHT, "Light"},
    {ATTRIBUTE_DARK, "Dark"},   {ATTRIBUTE_DEVINE, "Divine"},
};
DEFINE_X_TO_STRING_FUN(attribute_to_string, attribute2str)

static const ankerl::unordered_dense::map<uint8_t, uint8_t> attribute2id =
    make_ids(attribute2str);
DEFINE_X_TO_ID_FUN(attribute_to_id, attribute2id)


#define RACE_NONE 0x0 // token

static const std::map<uint32_t, std::string> race2str = {
    {RACE_NONE, "None"},
    {RACE_WARRIOR, "Warrior"},
    {RACE_SPELLCASTER, "Spellcaster"},
    {RACE_FAIRY, "Fairy"},
    {RACE_FIEND, "Fiend"},
    {RACE_ZOMBIE, "Zombie"},
    {RACE_MACHINE, "Machine"},
    {RACE_AQUA, "Aqua"},
    {RACE_PYRO, "Pyro"},
    {RACE_ROCK, "Rock"},
    {RACE_WINDBEAST, "Windbeast"},
    {RACE_PLANT, "Plant"},
    {RACE_INSECT, "Insect"},
    {RACE_THUNDER, "Thunder"},
    {RACE_DRAGON, "Dragon"},
    {RACE_BEAST, "Beast"},
    {RACE_BEASTWARRIOR, "Beast Warrior"},
    {RACE_DINOSAUR, "Dinosaur"},
    {RACE_FISH, "Fish"},
    {RACE_SEASERPENT, "Sea Serpent"},
    {RACE_REPTILE, "Reptile"},
    {RACE_PSYCHO, "Psycho"},
    {RACE_DEVINE, "Divine"},
    {RACE_CREATORGOD, "Creator God"},
    {RACE_WYRM, "Wyrm"},
    {RACE_CYBERSE, "Cyberse"},
    {RACE_ILLUSION, "Illusion"}};

static const ankerl::unordered_dense::map<uint32_t, uint8_t> race2id =
    make_ids(race2str);
DEFINE_X_TO_ID_FUN(race_to_id, race2id)


static const std::map<uint32_t, std::string> type2str = {
    {TYPE_MONSTER, "Monster"},
    {TYPE_SPELL, "Spell"},
    {TYPE_TRAP, "Trap"},
    {TYPE_NORMAL, "Normal"},
    {TYPE_EFFECT, "Effect"},
    {TYPE_FUSION, "Fusion"},
    {TYPE_RITUAL, "Ritual"},
    {TYPE_TRAPMONSTER, "Trap Monster"},
    {TYPE_SPIRIT, "Spirit"},
    {TYPE_UNION, "Union"},
    {TYPE_DUAL, "Dual"},
    {TYPE_TUNER, "Tuner"},
    {TYPE_SYNCHRO, "Synchro"},
    {TYPE_TOKEN, "Token"},
    {TYPE_QUICKPLAY, "Quick-play"},
    {TYPE_CONTINUOUS, "Continuous"},
    {TYPE_EQUIP, "Equip"},
    {TYPE_FIELD, "Field"},
    {TYPE_COUNTER, "Counter"},
    {TYPE_FLIP, "Flip"},
    {TYPE_TOON, "Toon"},
    {TYPE_XYZ, "XYZ"},
    {TYPE_PENDULUM, "Pendulum"},
    {TYPE_SPSUMMON, "Special"},
    {TYPE_LINK, "Link"},
};

inline std::vector<uint8_t> type_to_ids(uint32_t type) {
  std::vector<uint8_t> ids;
  ids.reserve(type2str.size());
  for (const auto &[k, v] : type2str) {
    ids.push_back(std::min(1u, type & k));
  }
  return ids;
}

static const std::map<int, std::string> phase2str = {
    {PHASE_DRAW, "draw phase"},
    {PHASE_STANDBY, "standby phase"},
    {PHASE_MAIN1, "main1 phase"},
    {PHASE_BATTLE_START, "battle start phase"},
    {PHASE_BATTLE_STEP, "battle step phase"},
    {PHASE_DAMAGE, "damage phase"},
    {PHASE_DAMAGE_CAL, "damage calculation phase"},
    {PHASE_BATTLE, "battle phase"},
    {PHASE_MAIN2, "main2 phase"},
    {PHASE_END, "end phase"},
};
DEFINE_X_TO_STRING_FUN(phase_to_string, phase2str)

static const ankerl::unordered_dense::map<int, uint8_t> phase2id =
    make_ids(phase2str);
DEFINE_X_TO_ID_FUN(phase_to_id, phase2id)


static const std::vector<int> _msgs = {
    MSG_SELECT_IDLECMD,  MSG_SELECT_CHAIN,     MSG_SELECT_CARD,
    MSG_SELECT_TRIBUTE,  MSG_SELECT_POSITION,  MSG_SELECT_EFFECTYN,
    MSG_SELECT_YESNO,    MSG_SELECT_BATTLECMD, MSG_SELECT_UNSELECT_CARD,
    MSG_SELECT_OPTION,   MSG_SELECT_PLACE,     MSG_SELECT_SUM,
    MSG_SELECT_DISFIELD, MSG_ANNOUNCE_ATTRIB,  MSG_ANNOUNCE_NUMBER,
    MSG_ANNOUNCE_CARD,   MSG_ANNOUNCE_RACE,
    // Append only: the frozen 40M message embedding keeps its old row IDs.
    MSG_SORT_CARD,      MSG_SELECT_COUNTER,    MSG_ROCK_PAPER_SCISSORS,
};

static const ankerl::unordered_dense::map<int, uint8_t> msg2id =
    make_ids(_msgs, 1);
DEFINE_X_TO_ID_FUN(msg_to_id, msg2id)


enum class ActionAct {
  None,
  Set,
  Repo,
  SpSummon,
  Summon,
  MSet,
  Attack,
  DirectAttack,
  Activate,
  Cancel,
};

inline std::string action_act_to_string(ActionAct act) {
  switch (act) {
  case ActionAct::None:
    return "None";
  case ActionAct::Set:
    return "Set";
  case ActionAct::Repo:
    return "Repo";
  case ActionAct::SpSummon:
    return "SpSummon";
  case ActionAct::Summon:
    return "Summon";
  case ActionAct::MSet:
    return "MSet";
  case ActionAct::Attack:
    return "Attack";
  case ActionAct::DirectAttack:
    return "DirectAttack";
  case ActionAct::Activate:
    return "Activate";
  case ActionAct::Cancel:
    return "Cancel";
  default:
    return "Unknown";
  }
}

enum class ActionPhase {
  None,
  Battle,
  Main2,
  End,
};

inline std::string action_phase_to_string(ActionPhase phase) {
  switch (phase) {
  case ActionPhase::None:
    return "None";
  case ActionPhase::Battle:
    return "Battle";
  case ActionPhase::Main2:
    return "Main2";
  case ActionPhase::End:
    return "End";
  default:
    return "Unknown";
  }
}

enum class ActionPlace {
  None,
  MZone1,
  MZone2,
  MZone3,
  MZone4,
  MZone5,
  MZone6,
  MZone7,
  SZone1,
  SZone2,
  SZone3,
  SZone4,
  SZone5,
  SZone6,
  SZone7,
  SZone8,
  OpMZone1,
  OpMZone2,
  OpMZone3,
  OpMZone4,
  OpMZone5,
  OpMZone6,
  OpMZone7,
  OpSZone1,
  OpSZone2,
  OpSZone3,
  OpSZone4,
  OpSZone5,
  OpSZone6,
  OpSZone7,
  OpSZone8,
};


inline std::vector<ActionPlace> flag_to_usable_places(
  uint32_t flag, bool reverse = false) {
  std::vector<ActionPlace> places;
  for (int j = 0; j < 4; j++) {
    uint32_t value = (flag >> (j * 8)) & 0xff;
    for (int i = 0; i < 8; i++) {
      bool avail = (value & (1 << i)) == 0;
      if (reverse) {
        avail = !avail;
      }
      if (avail) {
        ActionPlace place;
        if (j == 0) {
          place = static_cast<ActionPlace>(i + static_cast<int>(ActionPlace::MZone1));
        } else if (j == 1) {
          place = static_cast<ActionPlace>(i + static_cast<int>(ActionPlace::SZone1));
        } else if (j == 2) {
          place = static_cast<ActionPlace>(i + static_cast<int>(ActionPlace::OpMZone1));
        } else if (j == 3) {
          place = static_cast<ActionPlace>(i + static_cast<int>(ActionPlace::OpSZone1));
        }
        places.push_back(place);
      }
    }
  }
  return places;
}

inline std::array<uint8_t, 3> core_encode_place(ActionPlace place,
                                                uint8_t player) {
  const int value = static_cast<int>(place);
  uint8_t controller = player;
  uint8_t location = 0;
  uint8_t sequence = 0;
  if (value >= static_cast<int>(ActionPlace::MZone1) &&
      value <= static_cast<int>(ActionPlace::MZone7)) {
    location = LOCATION_MZONE;
    sequence = value - static_cast<int>(ActionPlace::MZone1);
  } else if (value >= static_cast<int>(ActionPlace::SZone1) &&
             value <= static_cast<int>(ActionPlace::SZone8)) {
    location = LOCATION_SZONE;
    sequence = value - static_cast<int>(ActionPlace::SZone1);
  } else if (value >= static_cast<int>(ActionPlace::OpMZone1) &&
             value <= static_cast<int>(ActionPlace::OpMZone7)) {
    controller = 1 - player;
    location = LOCATION_MZONE;
    sequence = value - static_cast<int>(ActionPlace::OpMZone1);
  } else if (value >= static_cast<int>(ActionPlace::OpSZone1) &&
             value <= static_cast<int>(ActionPlace::OpSZone8)) {
    controller = 1 - player;
    location = LOCATION_SZONE;
    sequence = value - static_cast<int>(ActionPlace::OpSZone1);
  } else {
    throw std::runtime_error("Invalid action place");
  }
  return {controller, location, sequence};
}

inline std::vector<uint32_t> core_exact_mask_options(uint32_t allowed,
                                                     int count,
                                                     int bits) {
  std::vector<uint32_t> values;
  std::vector<uint32_t> candidates;
  for (int bit = 0; bit < bits; ++bit) {
    const uint32_t value = uint32_t{1} << bit;
    if (allowed & value) candidates.push_back(value);
  }
  if (count < 0 || count > static_cast<int>(candidates.size())) {
    throw std::runtime_error("Invalid exact-mask selection count");
  }
  if (count == 0) return {0};
  for (const auto &combination : combinations(candidates.size(), count)) {
    uint32_t value = 0;
    for (int index : combination) value |= candidates[index];
    values.push_back(value);
  }
  return values;
}

inline size_t core_bounded_combination_count(int n, int k, size_t capacity) {
  if (k < 0 || k > n) return 0;
  k = std::min(k, n - k);
  size_t count = 1;
  for (int i = 1; i <= k; ++i) {
    const size_t numerator = static_cast<size_t>(n - k + i);
    if (count > std::numeric_limits<size_t>::max() / numerator) {
      return capacity + 1;
    }
    count = (count * numerator) / static_cast<size_t>(i);
    if (count > capacity) return capacity + 1;
  }
  return count;
}

inline std::tuple<std::vector<uint8_t>, bool, std::vector<uint8_t>>
core_announce_race_core_fixture(uint32_t available, uint8_t count,
                                uint32_t response) {
  duel fixture;
  fixture.game_field->core.units.emplace_back();
  fixture.game_field->announce_race(
      0, 0, count, static_cast<int32_t>(available));
  std::vector<uint8_t> request(fixture.message_buffer.begin(),
                               fixture.message_buffer.end());
  if (request.size() != 7 || request[0] != MSG_ANNOUNCE_RACE) {
    throw std::runtime_error("Malformed core announce-race request");
  }
  fixture.clear_buffer();
  fixture.set_responsei(response);
  bool accepted = fixture.game_field->announce_race(
      1, 0, request[2], static_cast<int32_t>(available));
  return {request, accepted,
          std::vector<uint8_t>(fixture.message_buffer.begin(),
                               fixture.message_buffer.end())};
}

inline std::vector<uint8_t> core_position_options(uint8_t allowed) {
  std::vector<uint8_t> positions;
  for (uint8_t position : {uint8_t(POS_FACEUP_ATTACK),
                           uint8_t(POS_FACEDOWN_ATTACK),
                           uint8_t(POS_FACEUP_DEFENSE),
                           uint8_t(POS_FACEDOWN_DEFENSE)}) {
    if (allowed & position) positions.push_back(position);
  }
  return positions;
}

inline std::array<bool, 3> core_automatic_selection_fixture() {
  duel card_fixture;
  const bool card_short = card_fixture.game_field->select_card(
      0, 0, 0, 0, 0) && card_fixture.message_buffer.empty();
  duel position_fixture;
  const bool position_short = position_fixture.game_field->select_position(
      0, 0, 1000, POS_FACEUP_ATTACK) &&
      position_fixture.message_buffer.empty() &&
      position_fixture.game_field->returns.ivalue[0] == POS_FACEUP_ATTACK;
  duel counter_fixture;
  const bool counter_short = counter_fixture.game_field->select_counter(
      0, 0, 1, 0, 1, 0) && counter_fixture.message_buffer.empty();
  return {card_short, position_short, counter_short};
}

inline std::vector<std::vector<uint8_t>> core_sort_responses(int count) {
  if (count < 0 || count > 8) {
    throw std::runtime_error("Sort response fixture count must be 0..8");
  }
  std::vector<std::vector<uint8_t>> responses;
  std::vector<uint8_t> order(count);
  std::iota(order.begin(), order.end(), uint8_t{0});
  do {
    responses.push_back(order);
  } while (std::next_permutation(order.begin(), order.end()));
  responses.push_back({uint8_t{0xff}});
  return responses;
}

inline std::tuple<std::vector<uint8_t>, bool, std::vector<uint8_t>>
core_sort_card_core_fixture(int count, const std::vector<uint8_t> &response) {
  if (count < 1 || count > 255 ||
      !(response.size() == static_cast<size_t>(count) ||
        (response.size() == 1 && response[0] == 0xff))) {
    throw std::runtime_error("Invalid sort-card fixture shape");
  }
  duel fixture;
  for (int index = 0; index < count; ++index) {
    card *pcard = fixture.new_card(0);
    pcard->data.code = 1000 + index;
    pcard->current.controler = 0;
    pcard->current.location = LOCATION_HAND;
    pcard->current.sequence = index;
    fixture.game_field->core.select_cards.push_back(pcard);
  }
  fixture.game_field->sort_card(0, 0);
  std::vector<uint8_t> request(fixture.message_buffer.begin(),
                               fixture.message_buffer.end());
  fixture.clear_buffer();
  std::array<uint8_t, SIZE_RETURN_VALUE> encoded{};
  std::copy(response.begin(), response.end(), encoded.begin());
  fixture.set_responseb(encoded.data());
  const bool accepted = fixture.game_field->sort_card(1, 0);
  return {request, accepted,
          std::vector<uint8_t>(fixture.message_buffer.begin(),
                               fixture.message_buffer.end())};
}

inline std::tuple<std::vector<uint8_t>, bool, std::vector<uint8_t>>
core_select_counter_core_fixture(const std::vector<uint16_t> &capacities,
                                 uint16_t requested,
                                 const std::vector<uint16_t> &allocation) {
  if (capacities.empty() || capacities.size() > 7 ||
      allocation.size() != capacities.size()) {
    throw std::runtime_error("Invalid counter fixture shape");
  }
  constexpr uint16_t counter_type = 1;
  duel fixture;
  for (int index = 0; index < static_cast<int>(capacities.size()); ++index) {
    card *pcard = fixture.new_card(0);
    pcard->data.code = 1000 + index;
    pcard->current.controler = 0;
    pcard->current.location = LOCATION_MZONE;
    pcard->current.sequence = index;
    pcard->counters[counter_type] = capacities[index];
    fixture.game_field->player[0].list_mzone[index] = pcard;
  }
  fixture.game_field->select_counter(0, 0, counter_type, requested, 1, 0);
  std::vector<uint8_t> request(fixture.message_buffer.begin(),
                               fixture.message_buffer.end());
  fixture.clear_buffer();
  std::array<uint8_t, SIZE_RETURN_VALUE> encoded{};
  for (int index = 0; index < static_cast<int>(allocation.size()); ++index) {
    encoded[index * 2] = static_cast<uint8_t>(allocation[index]);
    encoded[index * 2 + 1] = static_cast<uint8_t>(allocation[index] >> 8);
  }
  fixture.set_responseb(encoded.data());
  const bool accepted = fixture.game_field->select_counter(
      1, 0, counter_type, requested, 1, 0);
  return {request, accepted,
          std::vector<uint8_t>(fixture.message_buffer.begin(),
                               fixture.message_buffer.end())};
}

inline std::tuple<std::vector<uint8_t>, bool, std::vector<uint8_t>>
core_weighted_core_fixture(CoreWeightedKind kind,
                           const std::vector<uint32_t> &must,
                           const std::vector<uint32_t> &optional,
                           int32_t target, int min_count, int max_count,
                           const std::vector<int> &selected) {
  if (optional.empty() || optional.size() > 255 ||
      must.size() + selected.size() > 255 ||
      kind == CoreWeightedKind::SumLimit && !must.empty() &&
          must.size() > 255) {
    throw std::runtime_error("Invalid weighted core fixture shape");
  }
  duel fixture;
  fixture.game_field->core.units.emplace_back();
  auto add_card = [&](uint32_t value, uint8_t sequence) {
    card *pcard = fixture.new_card(0);
    pcard->data.code = 1000 + sequence;
    pcard->current.controler = 0;
    pcard->current.location = LOCATION_HAND;
    pcard->current.sequence = sequence;
    pcard->release_param = value;
    pcard->sum_param = value;
    return pcard;
  };
  if (kind != CoreWeightedKind::Tribute) {
    for (int index = 0; index < static_cast<int>(must.size()); ++index) {
      fixture.game_field->core.must_select_cards.push_back(
          add_card(must[index], static_cast<uint8_t>(index)));
    }
  }
  for (int index = 0; index < static_cast<int>(optional.size()); ++index) {
    fixture.game_field->core.select_cards.push_back(add_card(
        optional[index], static_cast<uint8_t>(must.size() + index)));
  }
  bool request_pending = false;
  if (kind == CoreWeightedKind::Tribute) {
    request_pending = !fixture.game_field->select_tribute(
        0, 0, 0, static_cast<uint8_t>(target),
        static_cast<uint8_t>(max_count));
  } else {
    request_pending = !fixture.game_field->select_with_sum_limit(
        0, 0, target, min_count,
        kind == CoreWeightedKind::ExactSum ? max_count : 0);
  }
  std::vector<uint8_t> request(fixture.message_buffer.begin(),
                               fixture.message_buffer.end());
  if (!request_pending || request.empty()) {
    throw std::runtime_error("Weighted core fixture did not emit a request");
  }
  fixture.clear_buffer();
  std::array<uint8_t, SIZE_RETURN_VALUE> response{};
  response[0] = static_cast<uint8_t>(must.size() + selected.size());
  for (int index = 0; index < static_cast<int>(selected.size()); ++index) {
    if (selected[index] < 0 || selected[index] >= static_cast<int>(optional.size())) {
      throw std::runtime_error("Weighted fixture index out of range");
    }
    response[must.size() + index + 1] = static_cast<uint8_t>(selected[index]);
  }
  fixture.set_responseb(response.data());
  bool accepted;
  if (kind == CoreWeightedKind::Tribute) {
    accepted = fixture.game_field->select_tribute(
        1, 0, 0, request[3], request[4]);
  } else {
    accepted = fixture.game_field->select_with_sum_limit(
        1, 0, target, min_count,
        kind == CoreWeightedKind::ExactSum ? max_count : 0);
  }
  return {request, accepted,
          std::vector<uint8_t>(fixture.message_buffer.begin(),
                               fixture.message_buffer.end())};
}

inline std::vector<uint8_t> core_index_response(
    const std::vector<int> &indices, int candidate_count) {
  if (indices.size() > 255) {
    throw std::runtime_error("Too many selected indices");
  }
  std::unordered_set<int> seen;
  std::vector<uint8_t> response;
  response.reserve(indices.size() + 1);
  response.push_back(static_cast<uint8_t>(indices.size()));
  for (int index : indices) {
    if (index < 0 || index >= candidate_count || !seen.insert(index).second) {
      throw std::runtime_error("Invalid or duplicate selected index");
    }
    response.push_back(static_cast<uint8_t>(index));
  }
  return response;
}

inline uint32_t core_command_response(uint16_t command, uint16_t index) {
  return static_cast<uint32_t>(command) |
         (static_cast<uint32_t>(index) << 16);
}

inline uint32_t core_uint32_identity(uint32_t value) { return value; }

inline int32_t core_scalar_response(int32_t value, int32_t minimum,
                                    int32_t maximum) {
  if (value < minimum || value > maximum) {
    throw std::runtime_error("Scalar response is outside the declared range");
  }
  return value;
}

inline void core_validate_action_capacity(size_t candidate_count,
                                         size_t action_capacity) {
  if (candidate_count > action_capacity) {
    throw std::runtime_error(fmt::format(
        "[protocol boundary] {} legal actions exceed model capacity {}",
        candidate_count, action_capacity));
  }
}

inline uint8_t core_select_unselect_response_index(size_t select_count,
                                                  size_t unselect_count,
                                                  bool unselect,
                                                  size_t candidate_index) {
  const size_t pool_size = unselect ? unselect_count : select_count;
  if (candidate_index >= pool_size || select_count + unselect_count > 256) {
    throw std::runtime_error(
        "[protocol boundary] invalid select/unselect response index");
  }
  return static_cast<uint8_t>(candidate_index + (unselect ? select_count : 0));
}

inline std::tuple<std::vector<uint8_t>, bool, std::vector<uint8_t>>
core_select_unselect_core_fixture(size_t select_count, size_t unselect_count,
                                  bool finishable, bool cancelable,
                                  int32_t response_index) {
  if (select_count > UINT8_MAX || unselect_count > UINT8_MAX ||
      select_count + unselect_count == 0) {
    throw std::runtime_error("Invalid select/unselect fixture pool sizes");
  }
  duel fixture;
  auto add_card = [&](uint32_t code, uint8_t sequence) {
    card *pcard = fixture.new_card(0);
    pcard->data.code = code;
    pcard->current.controler = 0;
    pcard->current.location = LOCATION_HAND;
    pcard->current.sequence = sequence;
    pcard->current.position = POS_FACEUP;
    return pcard;
  };
  for (size_t i = 0; i < select_count; ++i) {
    fixture.game_field->core.select_cards.push_back(
        add_card(1000 + static_cast<uint32_t>(i), static_cast<uint8_t>(i)));
  }
  for (size_t i = 0; i < unselect_count; ++i) {
    fixture.game_field->core.unselect_cards.push_back(
        add_card(2000 + static_cast<uint32_t>(i), static_cast<uint8_t>(i)));
  }
  fixture.game_field->select_unselect_card(
      0, 0, cancelable, 1, 1, finishable);
  std::vector<uint8_t> request(fixture.message_buffer.begin(),
                               fixture.message_buffer.end());
  fixture.clear_buffer();
  if (response_index == -1) {
    fixture.set_responsei(static_cast<uint32_t>(-1));
  } else if (response_index >= 0 && response_index <= UINT8_MAX) {
    std::array<uint8_t, SIZE_RETURN_VALUE> response{};
    response[0] = 1;
    response[1] = static_cast<uint8_t>(response_index);
    fixture.set_responseb(response.data());
  } else {
    throw std::runtime_error("Invalid select/unselect fixture response index");
  }
  bool accepted = fixture.game_field->select_unselect_card(
      1, 0, cancelable, 1, 1, finishable);
  return {request, accepted,
          std::vector<uint8_t>(fixture.message_buffer.begin(),
                               fixture.message_buffer.end())};
}

inline std::array<uint32_t, 3> core_rock_paper_scissors_options() {
  return {1, 2, 3};
}

inline uint8_t core_rock_paper_scissors_player(uint8_t player) {
  if (player > 1) {
    throw std::runtime_error(fmt::format(
        "Invalid MSG_ROCK_PAPER_SCISSORS player {}", player));
  }
  return player;
}

inline std::pair<std::vector<std::vector<uint8_t>>, int32_t>
core_rock_paper_scissors_fixture(const std::vector<int32_t> &responses,
                                 bool repeat) {
  duel fixture;
  fixture.game_field->add_process(PROCESSOR_ROCK_PAPER_SCISSORS, 0, nullptr,
                                  nullptr, repeat ? 1 : 0, 0);
  std::vector<std::vector<uint8_t>> frames;
  size_t response_index = 0;
  for (size_t step = 0; step < 128; ++step) {
    fixture.clear_buffer();
    fixture.game_field->process();
    if (fixture.message_buffer.empty()) {
      break;
    }
    frames.emplace_back(fixture.message_buffer.begin(),
                        fixture.message_buffer.end());
    if (fixture.message_buffer[0] == MSG_ROCK_PAPER_SCISSORS) {
      if (fixture.message_buffer.size() != 2 ||
          response_index >= responses.size()) {
        throw std::runtime_error("Incomplete RPS core fixture response");
      }
      auto player = core_rock_paper_scissors_player(
          fixture.message_buffer[1]);
      if (player != (response_index % 2)) {
        throw std::runtime_error("Unexpected RPS core request player order");
      }
      auto hand = core_scalar_response(responses[response_index++], 1, 3);
      fixture.set_responsei(static_cast<uint32_t>(hand));
    } else if (fixture.message_buffer[0] == MSG_HAND_RES) {
      if (fixture.message_buffer.size() != 2) {
        throw std::runtime_error("Malformed RPS core result frame");
      }
      if (fixture.game_field->core.units.empty()) {
        if (response_index != responses.size()) {
          throw std::runtime_error("Unused RPS core fixture responses");
        }
        return {frames, fixture.game_field->returns.ivalue[0]};
      }
    }
  }
  throw std::runtime_error("RPS core fixture did not reach terminal result");
}

inline std::string action_place_to_string(ActionPlace place) {
  int i = static_cast<int>(place);
  if (i == 0) {
    return "None";
  }
  else if (i >= static_cast<int>(ActionPlace::MZone1) && i <= static_cast<int>(ActionPlace::MZone7)) {
    return fmt::format("m{}", i - static_cast<int>(ActionPlace::MZone1) + 1);
  }
  else if (i >= static_cast<int>(ActionPlace::SZone1) && i <= static_cast<int>(ActionPlace::SZone8)) {
    return fmt::format("s{}", i - static_cast<int>(ActionPlace::SZone1) + 1);
  }
  else if (i >= static_cast<int>(ActionPlace::OpMZone1) && i <= static_cast<int>(ActionPlace::OpMZone7)) {
    return fmt::format("om{}", i - static_cast<int>(ActionPlace::OpMZone1) + 1);
  }
  else if (i >= static_cast<int>(ActionPlace::OpSZone1) && i <= static_cast<int>(ActionPlace::OpSZone8)) {
    return fmt::format("os{}", i - static_cast<int>(ActionPlace::OpSZone1) + 1);
  }
  else {
    return "Unknown";
  }
}


inline std::pair<uint8_t, uint8_t> float_transform(int x) {
  x = x % 65536;
  return {
      static_cast<uint8_t>(x >> 8),
      static_cast<uint8_t>(x & 0xff),
  };
}

static std::vector<int> find_substrs(const std::string &str,
                                     const std::string &substr) {
  std::vector<int> res;
  int pos = 0;
  while ((pos = str.find(substr, pos)) != std::string::npos) {
    res.push_back(pos);
    pos += substr.length();
  }
  return res;
}

inline std::string time_now() {
  // strftime %Y-%m-%d %H-%M-%S
  time_t now = time(0);
  tm *ltm = localtime(&now);
  char buffer[80];
  strftime(buffer, 80, "%Y-%m-%d %H-%M-%S", ltm);
  return std::string(buffer);
}

// from ygopro/gframe/replay.h

// replay flag
#define REPLAY_COMPRESSED	0x1
#define REPLAY_TAG			0x2
#define REPLAY_DECODED		0x4
#define REPLAY_SINGLE_MODE	0x8
#define REPLAY_UNIFORM		0x10

// max size
#define MAX_REPLAY_SIZE	0x20000


struct ReplayHeader {
	unsigned int id;
	unsigned int version;
	unsigned int flag;
	unsigned int seed;
	unsigned int datasize;
	unsigned int start_time;
	unsigned char props[8];

	ReplayHeader()
		: id(0), version(0), flag(0), seed(0), datasize(0), start_time(0), props{ 0 } {}
};

// from ygopro/gframe/replay.h

using PlayerId = uint8_t;
using CardCode = uint32_t;
using CardId = uint16_t;

const int DESCRIPTION_LIMIT = 10000;
const int CARD_EFFECT_OFFSET = 10010;

class LegalAction {
public:
  std::string spec_ = "";
  bool unselect_ = false;
  ActionAct act_ = ActionAct::None;
  ActionPhase phase_ = ActionPhase::None;
  bool finish_ = false;
  uint8_t position_ = 0;
  int effect_ = -1;
  uint32_t number_ = 0;
  ActionPlace place_ = ActionPlace::None;
  uint8_t attribute_ = 0;
  uint32_t race_ = 0;

  int spec_index_ = 0;
  CardId cid_ = 0;
  int msg_ = 0;
  uint32_t response_ = 0;

  static LegalAction from_spec(const std::string &spec) {
    LegalAction la;
    la.spec_ = spec;
    return la;
  }

  static LegalAction act_spec(ActionAct act, const std::string &spec) {
    LegalAction la;
    la.act_ = act;
    la.spec_ = spec;
    return la;
  }

  static LegalAction finish() {
    LegalAction la;
    la.finish_ = true;
    return la;
  }

  static LegalAction cancel() {
    LegalAction la;
    la.act_ = ActionAct::Cancel;
    return la;
  }

  static LegalAction activate_spec(int effect_idx, const std::string &spec) {
    LegalAction la;
    la.act_ = ActionAct::Activate;
    la.effect_ = effect_idx;
    la.spec_ = spec;
    return la;
  }

  static LegalAction phase(ActionPhase phase) {
    LegalAction la;
    la.phase_ = phase;
    return la;
  }

  static LegalAction number(uint32_t number) {
    LegalAction la;
    la.number_ = number;
    return la;
  }

  static LegalAction place(ActionPlace place) {
    LegalAction la;
    la.place_ = place;
    return la;
  }

  static LegalAction attribute(int attribute) {
    LegalAction la;
    la.attribute_ = attribute;
    return la;
  }

  static LegalAction race(uint32_t race) {
    LegalAction la;
    la.race_ = race;
    la.response_ = race;
    return la;
  }
};

class SpecInfo {
public:
  uint16_t index;
  CardId cid;
};

class Card {
  friend class YGOProEnvImpl;

protected:
  CardCode code_ = 0;
  uint32_t alias_;
  uint64_t setcode_;
  uint32_t type_;
  uint32_t level_;
  uint32_t lscale_;
  uint32_t rscale_;
  int32_t attack_;
  int32_t defense_;
  uint32_t race_;
  uint32_t attribute_;
  uint32_t link_marker_;
  // uint32_t category_;
  std::string name_;
  std::string desc_;
  std::vector<std::string> strings_;

  uint32_t data_ = 0;

  uint32_t status_ = 0;
  PlayerId controler_ = 0;
  uint32_t location_ = 0;
  uint32_t sequence_ = 0;
  uint32_t position_ = 0;
  uint32_t counter_ = 0;

public:
  Card() = default;

  Card(CardCode code, uint32_t alias, uint64_t setcode, uint32_t type,
       uint32_t level, uint32_t lscale, uint32_t rscale, int32_t attack,
       int32_t defense, uint32_t race, uint32_t attribute, uint32_t link_marker,
       const std::string &name, const std::string &desc,
       const std::vector<std::string> &strings)
      : code_(code), alias_(alias), setcode_(setcode), type_(type),
        level_(level), lscale_(lscale), rscale_(rscale), attack_(attack),
        defense_(defense), race_(race), attribute_(attribute),
        link_marker_(link_marker), name_(name), desc_(desc), strings_(strings) {
  }

  ~Card() = default;

  void set_location(uint32_t location) {
    controler_ = location & 0xff;
    location_ = (location >> 8) & 0xff;
    sequence_ = (location >> 16) & 0xff;
    position_ = (location >> 24) & 0xff;
  }

  const std::string &name() const { return name_; }
  const std::string &desc() const { return desc_; }
  const uint32_t &type() const { return type_; }
  const uint32_t &level() const { return level_; }
  const std::vector<std::string> &strings() const { return strings_; }

  std::string get_spec(bool opponent) const {
    return ls_to_spec(location_, sequence_, position_, opponent);
  }

  std::string get_spec(PlayerId player) const {
    return get_spec(player != controler_);
  }

  std::string get_position() const { return position_to_string(position_); }

  std::string get_effect_description(CardCode code, int effect_idx) const {
    if (code == 0) {
      return get_system_string(effect_idx);
    }
    if (effect_idx == 0) {
      return "default";
    }
    effect_idx -= CARD_EFFECT_OFFSET;
    if (effect_idx < 0) {
      throw std::runtime_error(
          fmt::format("Invalid effect index: {}", effect_idx));
    }
    auto s = strings_[effect_idx];
    if (s.empty()) {
      return "effect " + std::to_string(effect_idx);
    }
    return s;
  }
};

struct MDuel {
  intptr_t pduel;
  uint64_t seed;
  std::vector<CardCode> main_deck0;
  std::vector<CardCode> extra_deck0;
  std::string deck_name0;
  std::vector<CardCode> main_deck1;
  std::vector<CardCode> extra_deck1;
  std::string deck_name1;
};

inline Card db_query_card(const SQLite::Database &db, CardCode code) {
  SQLite::Statement query1(db, "SELECT * FROM datas WHERE id=?");
  query1.bind(1, code);
  bool found = query1.executeStep();
  if (!found) {
    std::string msg = "[db_query_card] Card not found: " + std::to_string(code);
    throw std::runtime_error(msg);
  }

  uint32_t alias = query1.getColumn("alias");
  uint64_t setcode = query1.getColumn("setcode").getInt64();
  uint32_t type = query1.getColumn("type");
  uint32_t level_ = query1.getColumn("level");
  uint32_t level = level_ & 0xff;
  uint32_t lscale = (level_ >> 24) & 0xff;
  uint32_t rscale = (level_ >> 16) & 0xff;
  int32_t attack = query1.getColumn("atk");
  int32_t defense = query1.getColumn("def");
  uint32_t link_marker = 0;
  if (type & TYPE_LINK) {
    defense = 0;
    link_marker = defense;
  }
  uint32_t race = query1.getColumn("race");
  uint32_t attribute = query1.getColumn("attribute");

  SQLite::Statement query2(db, "SELECT * FROM texts WHERE id=?");
  query2.bind(1, code);
  query2.executeStep();

  std::string name = query2.getColumn(1);
  std::string desc = query2.getColumn(2);
  std::vector<std::string> strings;
  for (int i = 3; i < query2.getColumnCount(); ++i) {
    std::string str = query2.getColumn(i);
    strings.push_back(str);
  }
  return Card(code, alias, setcode, type, level, lscale, rscale, attack,
              defense, race, attribute, link_marker, name, desc, strings);
}

inline card_data db_query_card_data(const SQLite::Database &db, CardCode code) {
  SQLite::Statement query(db, "SELECT * FROM datas WHERE id=?");
  query.bind(1, code);
  query.executeStep();
  card_data card;
  card.code = code;
  card.alias = query.getColumn("alias");
  uint64_t setcode = query.getColumn("setcode").getInt64();
  card.set_setcode(setcode);
  card.type = query.getColumn("type");
  uint32_t level_ = query.getColumn("level");
  card.level = level_ & 0xff;
  card.lscale = (level_ >> 24) & 0xff;
  card.rscale = (level_ >> 16) & 0xff;
  card.attack = query.getColumn("atk");
  card.defense = query.getColumn("def");
  if (card.type & TYPE_LINK) {
    card.link_marker = card.defense;
    card.defense = 0;
  } else {
    card.link_marker = 0;
  }
  card.race = query.getColumn("race");
  card.attribute = query.getColumn("attribute");
  return card;
}

struct card_script {
  byte *buf;
  int len;
};

static ankerl::unordered_dense::map<CardCode, Card> cards_;
static ankerl::unordered_dense::map<CardCode, CardId> card_ids_;
static ankerl::unordered_dense::map<CardCode, card_data> cards_data_;
static ankerl::unordered_dense::map<std::string, card_script> cards_script_;
static ankerl::unordered_dense::map<std::string, std::vector<CardCode>>
    main_decks_;
static ankerl::unordered_dense::map<std::string, std::vector<CardCode>>
    extra_decks_;
static std::vector<std::string> deck_names_;
static ankerl::unordered_dense::map<std::string, int> deck_names_ids_;

inline const Card &c_get_card(CardCode code) {
  if (code == 0) {
    // The core uses code zero for an identity-hidden or empty card slot.
    // Preserve that unknown identity instead of treating it as a database miss.
    static const Card unknown_card{};
    return unknown_card;
  }
  auto it = cards_.find(code);
  if (it != cards_.end()) {
    return it->second;
  }
  throw std::runtime_error("[c_get_card] Card not found: " + std::to_string(code));
}

inline CardId &c_get_card_id(CardCode code) {
  if (code == 0) {
    static CardId unknown_id = 0;
    return unknown_id;
  }
  auto it = card_ids_.find(code);
  if (it != card_ids_.end()) {
    return it->second;
  }
  throw std::runtime_error("[c_get_card_id] Card not found: " + std::to_string(code));
}

inline void sort_extra_deck(std::vector<CardCode> &deck) {
  std::vector<CardCode> c;
  std::vector<std::pair<CardCode, int>> fusion, xyz, synchro, link;

  for (auto code : deck) {
    const Card &cc = c_get_card(code);
    if (cc.type() & TYPE_FUSION) {
      fusion.push_back({code, cc.level()});
    } else if (cc.type() & TYPE_XYZ) {
      xyz.push_back({code, cc.level()});
    } else if (cc.type() & TYPE_SYNCHRO) {
      synchro.push_back({code, cc.level()});
    } else if (cc.type() & TYPE_LINK) {
      link.push_back({code, cc.level()});
    } else {
      throw std::runtime_error("Not extra deck card");
    }
  }

  auto cmp = [](const std::pair<CardCode, int> &a,
                const std::pair<CardCode, int> &b) {
    return a.second < b.second;
  };
  std::sort(fusion.begin(), fusion.end(), cmp);
  std::sort(xyz.begin(), xyz.end(), cmp);
  std::sort(synchro.begin(), synchro.end(), cmp);
  std::sort(link.begin(), link.end(), cmp);

  for (const auto &tc : fusion) {
    c.push_back(tc.first);
  }
  for (const auto &tc : xyz) {
    c.push_back(tc.first);
  }
  for (const auto &tc : synchro) {
    c.push_back(tc.first);
  }
  for (const auto &tc : link) {
    c.push_back(tc.first);
  }

  deck = c;
}

inline void preload_deck(const SQLite::Database &db,
                         const std::vector<CardCode> &deck) {
  for (const auto &code : deck) {
    auto it = cards_.find(code);
    if (it == cards_.end()) {
      cards_[code] = db_query_card(db, code);
      if (card_ids_.find(code) == card_ids_.end()) {
        throw std::runtime_error("Card not found in code list: " +
                                 std::to_string(code));
      }
    }

    auto it2 = cards_data_.find(code);
    if (it2 == cards_data_.end()) {
      cards_data_[code] = db_query_card_data(db, code);
    }
  }
}

inline uint32 card_reader_callback(CardCode code, card_data *card) {
  auto it = cards_data_.find(code);
  if (it == cards_data_.end()) {
    // These three rule-only identities are the canonical second names for
    // Timaeus, Critias, and Hermos.  Current official-only cards.cdb builds do
    // not contain their legacy unofficial rows, while both the pinned core and
    // canonical card scripts still query their set code during rule checks.
    // They are never drawable cards and therefore must not be added to the
    // model code list or embedding table.  Mirror the established
    // cards-unofficial.cdb metadata (setcode 0x00a1, TYPE_TOKEN) locally.
    if (code == 10000050 || code == 10000060 || code == 10000070) {
      card_data supplemental{};
      supplemental.code = code;
      supplemental.set_setcode(0x00a1);
      supplemental.type = TYPE_TOKEN;
      *card = supplemental;
      return 0;
    }
    fmt::println("[card_reader_callback] Card not found: " + std::to_string(code));
    throw std::runtime_error("[card_reader_callback] Card not found: " + std::to_string(code));
  }
  *card = it->second;
  return 0;
}

inline byte *read_card_script(const std::string &path, int *lenptr) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    *lenptr = 0;
    return nullptr;
  }
  file.seekg(0, std::ios::end);
  int len = file.tellg();
  file.seekg(0, std::ios::beg);
  byte *buf = new byte[len];
  file.read((char *)buf, len);
  *lenptr = len;
  return buf;
}

inline byte *script_reader_callback(const char *name, int *lenptr) {
  std::string path(name);
  auto it = cards_script_.find(path);
  if (it == cards_script_.end()) {
    fmt::println("[script_reader_callback] Script not found: " + path);
    throw std::runtime_error("[script_reader_callback] Script not found: " + path);
  }
  *lenptr = it->second.len;
  return it->second.buf;
}

static void init_module(const std::string &db_path,
                        const std::string &code_list_file,
                        const std::map<std::string, std::string> &decks) {
  // parse code from code_list_file
  SQLite::Database db(db_path, SQLite::OPEN_READONLY);

  auto start = std::chrono::steady_clock::now();

  std::ifstream file(code_list_file);
  std::string line;
  int i = 0;
  CardCode code;
  int has_script, script_len;
  while (std::getline(file, line)) {
    i++;
    std::istringstream iss(line);
    if (!(iss >> code >> has_script)) {
        std::cerr << "Failed to parse line in code_list: " << line << std::endl;
        continue;
    }
    card_ids_[code] = i;
    cards_[code] = db_query_card(db, code);
    cards_data_[code] = db_query_card_data(db, code);
    if (has_script) {
      std::string path = "./script/c" + std::to_string(code) + ".lua";
      byte *buf = read_card_script(path, &script_len);
      cards_script_[path] = {buf, script_len};
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  // fmt::println("load {} cards in {}ms", cards_data_.size(), milliseconds);

  for (const auto &[name, deck] : decks) {
    auto [main_deck, extra_deck, side_deck] = read_decks(deck);
    main_decks_[name] = main_deck;
    extra_decks_[name] = extra_deck;
    if (name[0] != '_') {
      deck_names_.push_back(name);
      deck_names_ids_[name] = deck_names_.size() - 1;
    }
  }

  for (auto &[name, deck] : extra_decks_) {
    sort_extra_deck(deck);
  }

  card_data card;
  cards_data_[0] = card;

  std::vector<std::string> preload = {
    "./script/constant.lua",
    "./script/utility.lua",
    "./script/procedure.lua",
  };
  for (const auto &path : preload) {
    byte *buf = read_card_script(path, &script_len);
    cards_script_[path] = {buf, script_len};
  }
  cards_script_["./script/c0.lua"] = {nullptr, 0};

  set_card_reader(card_reader_callback);
  set_script_reader(script_reader_callback);
}

inline std::string getline() {
#ifdef _WIN32
  std::string input;
  if (std::getline(std::cin, input)) {
    return input;
  }
  exit(0);
#else
  char *line = nullptr;
  size_t len = 0;
  ssize_t read;

  read = getline(&line, &len, stdin);

  if (read != -1) {
    // Remove line ending character(s)
    if (line[read - 1] == '\n')
      line[read - 1] = '\0'; // Replace newline character with null terminator
    else if (line[read - 2] == '\r' && line[read - 1] == '\n') {
      line[read - 2] = '\0'; // Replace carriage return and newline characters
                             // with null terminator
      line[read - 1] = '\0';
    }

    std::string input(line);
    free(line);
    return input;
  } else {
    exit(0);
  }

  free(line);
  return "";
#endif
}

class Player {
  friend class YGOProEnvImpl;

protected:
  const std::string nickname_;
  const int init_lp_;
  const PlayerId duel_player_;
  const bool verbose_;

  bool seen_waiting_ = false;

public:
  Player(const std::string &nickname, int init_lp, PlayerId duel_player,
         bool verbose = false)
      : nickname_(nickname), init_lp_(init_lp), duel_player_(duel_player),
        verbose_(verbose) {}
  virtual ~Player() = default;

  void notify(const std::string &text) {
    if (verbose_) {
      fmt::println("{} {}", duel_player_, text);
    }
  }

  const int &init_lp() const { return init_lp_; }

  virtual int think(const std::vector<LegalAction> &actions) = 0;
};

class FirstActionAI : public Player {
public:
  FirstActionAI(const std::string &nickname, int init_lp, PlayerId duel_player,
                bool verbose = false)
      : Player(nickname, init_lp, duel_player, verbose) {}

  int think(const std::vector<LegalAction> &actions) override { return 0; }
};

class GreedyAI : public Player {
public:
  GreedyAI(const std::string &nickname, int init_lp, PlayerId duel_player,
           bool verbose = false)
      : Player(nickname, init_lp, duel_player, verbose) {}

  static int action_score(const LegalAction &action) {
    if (action.act_ == ActionAct::DirectAttack) return 1000;
    if (action.act_ == ActionAct::Attack) return 900;
    if (action.phase_ == ActionPhase::Battle) return 800;
    if (action.phase_ == ActionPhase::Main2) return 700;
    if (action.phase_ == ActionPhase::End) return 650;
    if (action.act_ == ActionAct::SpSummon) return 600;
    if (action.act_ == ActionAct::Summon) return 590;
    if (action.act_ == ActionAct::Activate) return 580;
    if (action.act_ == ActionAct::Set) return 500;
    if (action.act_ == ActionAct::MSet) return 490;
    if (action.act_ == ActionAct::Repo) return 400;
    if (action.finish_) return 300;
    if (action.act_ == ActionAct::Cancel) return 200;
    return 350;
  }

  int think(const std::vector<LegalAction> &actions) override {
    if (actions.empty()) {
      throw std::runtime_error("GreedyAI received no legal actions");
    }
    int best = 0;
    int best_score = action_score(actions[0]);
    for (int i = 1; i < static_cast<int>(actions.size()); ++i) {
      const int score = action_score(actions[i]);
      if (score > best_score) {
        best = i;
        best_score = score;
      }
    }
    return best;
  }
};

class RandomAI : public Player {
protected:
  std::mt19937 gen_;
  std::uniform_int_distribution<int> dist_;

public:
  RandomAI(int max_options, int seed, const std::string &nickname, int init_lp,
           PlayerId duel_player, bool verbose = false)
      : Player(nickname, init_lp, duel_player, verbose), gen_(seed),
        dist_(0, max_options - 1) {}

  int think(const std::vector<LegalAction> &actions) override {
    return dist_(gen_) % actions.size();
  }
};

class HumanPlayer : public Player {
protected:
public:
  HumanPlayer(const std::string &nickname, int init_lp, PlayerId duel_player,
              bool verbose = false)
      : Player(nickname, init_lp, duel_player, verbose) {}

  int think(const std::vector<LegalAction> &actions) override {
    while (true) {
      std::string input = getline();
      if (input == "quit") {
        exit(0);
      }
      int idx = -1;
      try {
        idx = std::stoi(input) - 1;
      } catch (std::invalid_argument &e) {
        fmt::println("{} Invalid input: {}", duel_player_, input);
        continue;
      }
      if (idx >= 0 && idx < actions.size()) {
        return idx;
      } else {
        fmt::println("{} Choose from {} actions", duel_player_, actions.size());
      }
    }
  }
};

class YGOProEnvFns {
public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("deck1"_.Bind(std::string("OldSchool")),
                    "deck2"_.Bind(std::string("OldSchool")), "player"_.Bind(-1),
                    "play_mode"_.Bind(std::string("bot")),
                    "verbose"_.Bind(false), "max_options"_.Bind(16),
                    "max_cards"_.Bind(80), "n_history_actions"_.Bind(16),
                    "record"_.Bind(false), "async_reset"_.Bind(false),
                    "greedy_reward"_.Bind(true), "timeout"_.Bind(600),
                    "oppo_info"_.Bind(false), "max_steps"_.Bind(1000),
                    "observation_schema"_.Bind(std::string("legacy-v2")),
                    "n_public_events"_.Bind(32),
                    "max_group_references"_.Bind(8),
                    "semantic_asset_dir"_.Bind(std::string("")),
                    "windbot_host"_.Bind(std::string("127.0.0.1")),
                    "windbot_port"_.Bind(0), "windbot_timeout"_.Bind(30),
                    "deck_sampling_manifest"_.Bind(std::string("")),
                    "deck_sampler_seed"_.Bind(std::string("0")),
                    "deck_sampler_counters"_.Bind(std::string("")));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config &conf) {
    int n_action_feats = 12;
    return MakeDict(
        "obs:cards_"_.Bind(Spec<uint8_t>({conf["max_cards"_] * 2, 41})),
        "obs:global_"_.Bind(Spec<uint8_t>({23})),
        "obs:actions_"_.Bind(
            Spec<uint8_t>({conf["max_options"_], n_action_feats})),
        "obs:h_actions_"_.Bind(
            Spec<uint8_t>({conf["n_history_actions"_], n_action_feats + 2})),
        "obs:mask_"_.Bind(Spec<uint8_t>({conf["max_cards"_] * 2, 14})),
        // Structured-lite tensors are present in the versioned native artifact.
        // The Python schema wrapper removes them for legacy-v2 callers.
        "obs:visible_card_ids_"_.Bind(
            Spec<uint8_t>({conf["max_cards"_] * 2, 2})),
        "obs:card_semantics_"_.Bind(
            Spec<uint8_t>({conf["max_cards"_] * 2, 32})),
        "obs:effect_tags_"_.Bind(
            Spec<uint8_t>({conf["max_cards"_] * 2, 16})),
        "obs:effect_tag_confidence_"_.Bind(
            Spec<uint8_t>({conf["max_cards"_] * 2, 16})),
        "obs:selection_"_.Bind(Spec<uint8_t>({16})),
        "obs:action_features_"_.Bind(
            Spec<uint8_t>({conf["max_options"_], 16})),
        "obs:action_single_refs_"_.Bind(
            Spec<uint16_t>({conf["max_options"_], 4, 3})),
        "obs:action_group_refs_"_.Bind(
            Spec<uint16_t>({conf["max_options"_], 4,
                            conf["max_group_references"_], 2})),
        "obs:action_group_mask_"_.Bind(
            Spec<uint8_t>({conf["max_options"_], 4,
                           conf["max_group_references"_]})),
        "obs:public_events_"_.Bind(
            Spec<uint8_t>({conf["n_public_events"_], 12})),
        "obs:public_event_refs_"_.Bind(
            Spec<uint16_t>({conf["n_public_events"_], 4, 3})),
        "obs:structured_diagnostics_"_.Bind(Spec<uint8_t>({8})),
        "info:num_options"_.Bind(Spec<int>({}, {0, conf["max_options"_]})),
        "info:to_play"_.Bind(Spec<int>({}, {0, 1})),
        "info:is_selfplay"_.Bind(Spec<int>({}, {0, 1})),
        "info:win_reason"_.Bind(Spec<int>({}, {-1, 1})),
        "info:step_time"_.Bind(Spec<double>({2})),
        "info:deck"_.Bind(Spec<int>({2})),
        "info:deck_cluster"_.Bind(Spec<int>({2})),
        "info:deck_family"_.Bind(Spec<int>({2})),
        "info:deck_member"_.Bind(Spec<int>({2})),
        "info:deck_is_anchor"_.Bind(Spec<int>({2}, {0, 1})),
        "info:deck_sampler_counter"_.Bind(Spec<int64_t>({2})),
        "info:invalid_game"_.Bind(Spec<int>({}, {0, 1})),
        "info:termination_reason"_.Bind(Spec<int>({}, {0, 3})),
        "info:episode_steps"_.Bind(Spec<int>({}, {0, conf["max_steps"_]})),
        "info:turn_count"_.Bind(Spec<int>({}, {0, conf["max_steps"_]}))
      );
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config &conf) {
    return MakeDict(
        "action"_.Bind(Spec<int>({}, {0, conf["max_options"_] - 1})));
  }
};

using YGOProEnvSpec = EnvSpec<YGOProEnvFns>;

enum PlayMode {
  kHuman, kSelfPlay, kRandomBot, kGreedyBot, kFirstActionBot, kWindBot, kCount
};

enum TerminationReason {
  kTerminationNone = 0,
  kTerminationNatural = 1,
  kTerminationMaxSteps = 2,
  kTerminationTimeout = 3,
};

// parse play modes seperated by '+'
inline std::vector<PlayMode> parse_play_modes(const std::string &play_mode) {
  std::vector<PlayMode> modes;
  std::istringstream ss(play_mode);
  std::string token;
  while (std::getline(ss, token, '+')) {
    if (token == "human") {
      modes.push_back(kHuman);
    } else if (token == "self") {
      modes.push_back(kSelfPlay);
    } else if (token == "bot") {
      modes.push_back(kGreedyBot);
    } else if (token == "first") {
      modes.push_back(kFirstActionBot);
    } else if (token == "random") {
      modes.push_back(kRandomBot);
    } else if (token == "windbot") {
      modes.push_back(kWindBot);
    } else {
      throw std::runtime_error("Unknown play mode: " + token);
    }
  }
  // human mode can't be combined with other modes
  if (std::find(modes.begin(), modes.end(), kHuman) != modes.end() &&
      modes.size() > 1) {
    throw std::runtime_error("Human mode can't be combined with other modes");
  }
  return modes;
}

// rules = 1, Traditional
// rules = 0, Default
// rules = 4, Link
// rules = 5, MR5
constexpr int32_t rules_ = 5;
constexpr int32_t duel_options_ = ((rules_ & 0xFF) << 16) + (0 & 0xFFFF);


class YGOProEnvImpl {
  friend class ProtocolAdapterProbe;
protected:
  const EnvSpec<YGOProEnvFns> spec_;
  const int env_id_;

  constexpr static int init_lp_ = 8000;
  constexpr static int startcount_ = 5;
  constexpr static int drawcount_ = 1;

  const std::string deck1_;
  const std::string deck2_;

  std::vector<uint32> main_deck0_;
  std::vector<uint32> main_deck1_;
  std::vector<uint32> extra_deck0_;
  std::vector<uint32> extra_deck1_;

  std::string deck_name_[2] = {"", ""};
  struct SamplerDeck {
    std::string deck_id;
    int deck_index;
  };
  struct SamplerFamily {
    std::string family_id;
    int family_index;
    std::vector<SamplerDeck> decks;
  };
  struct SamplerCluster {
    std::string cluster_id;
    int cluster_index;
    std::vector<SamplerFamily> families;
  };
  struct SamplerSelection {
    int cluster_index{-2};
    int family_index{-2};
    int deck_index{-2};
    int is_anchor{0};
    int64_t counter{0};
  };
  std::vector<SamplerCluster> sampler_clusters_;
  std::string sampler_anchor_deck_;
  int sampler_reserve_ppm_{0};
  uint64_t sampler_seed_{0};
  uint64_t sampler_counter_{0};
  SamplerSelection sampler_selection_[2];
  std::string nickname_[2] = {"Alice", "Bob"};

  const std::vector<PlayMode> play_modes_;

  // if play_mode_ == 'bot' or 'human', player_ is the order of the ai player
  // -1 means random, 0 and 1 means the first and second player respectively
  const int player_;

  PlayMode play_mode_;
  bool verbose_ = false;

  PlayerId ai_player_;

  intptr_t pduel_ = 0;
  std::unique_ptr<Player> players_[2]; //  abstract class must be pointer

  std::uniform_int_distribution<uint64_t> dist_int_;
  bool done_{true};
  long step_count_{0};
  bool invalid_game_{false};
  TerminationReason termination_reason_{kTerminationNone};
  bool duel_started_{false};
  uint32_t eng_flag_{0};

  PlayerId winner_;
  uint8_t win_reason_;
  const bool greedy_reward_;

  int lp_[2];

  // turn player
  PlayerId tp_ = 0;
  int current_phase_ = 0;
  int turn_count_;

  int msg_;
  std::vector<LegalAction> legal_actions_;
  PlayerId to_play_;
  std::function<void(int)> callback_;

  byte data_[4096];
  int dp_ = 0;
  int dl_ = 0;

  byte query_buf_[4096];
  int qdp_ = 0;

  byte resp_buf_[SIZE_RETURN_VALUE];

  int windbot_server_fd_ = -1;
  int windbot_fd_ = -1;
  const std::string windbot_host_;
  const int windbot_port_;
  const int windbot_timeout_;
  bool windbot_lobby_ready_ = false;

  using IdleCardSpec = std::tuple<CardCode, std::string, uint32_t>;

  // chain
  PlayerId chaining_player_;
  int chain_depth_ = 0;
  struct VisibleCardRef {
    CardCode code = 0;
    PlayerId controller = 0;
    uint8_t location = 0;
    uint8_t sequence = 0;
    uint8_t position = 0;
    bool present = false;
  };
  VisibleCardRef active_chain_source_;

  enum StructuredEventType : uint8_t {
    kEventNone = 0, kEventActivation = 1, kEventTarget = 2,
    kEventNegation = 3, kEventDestruction = 4, kEventMovement = 5,
    kEventDraw = 6, kEventSearch = 7, kEventSummon = 8,
    kEventSpecialSummon = 9, kEventChainSolving = 10,
    kEventChainSolved = 11, kEventChainEnd = 12,
  };
  struct PublicEventRecord {
    uint8_t type = 0;
    PlayerId actor = 0;
    uint8_t chain_link = 0;
    uint8_t location_from = 0;
    uint8_t location_to = 0;
    uint8_t position = 0;
    uint8_t effect = 0;
    uint8_t result = 0;
    int turn = 0;
    int phase = 0;
    VisibleCardRef card;
  };
  std::vector<PublicEventRecord> public_events_;
  uint32_t public_event_overflow_ = 0;

  std::vector<uint8_t> card_semantics_table_;
  std::vector<uint8_t> effect_tags_table_;
  std::vector<uint8_t> effect_tag_confidence_table_;

  bool selection_forced_ = false;
  bool selection_finishable_ = false;
  bool selection_cancelable_ = false;
  bool policy_back_cancel_suppressed_ = false;
  bool last_action_overflow_ = false;

  const int n_history_actions_;

  // circular buffer for history actions
  TArray<uint8_t> history_actions_1_;
  TArray<uint8_t> history_actions_2_;
  int ha_p_1_ = 0;
  int ha_p_2_ = 0;

  std::unordered_set<std::string> revealed_;

  // multi select
  int ms_idx_ = -1;
  int ms_mode_ = 0;
  int ms_min_ = 0;
  int ms_max_ = 0;
  int ms_must_ = 0;
  std::vector<std::string> ms_specs_;
  std::vector<std::vector<int>> ms_combs_;
  CoreWeightedKind ms_weighted_kind_ = CoreWeightedKind::Tribute;
  std::vector<uint32_t> ms_weighted_must_;
  std::vector<uint32_t> ms_weighted_optional_;
  int32_t ms_weighted_target_ = 0;
  ankerl::unordered_dense::map<std::string, int> ms_spec2idx_;
  std::vector<int> ms_r_idxs_;
  std::vector<int> ms_counter_capacities_;
  std::vector<uint16_t> ms_counter_allocations_;
  int ms_counter_remaining_ = 0;
  int ms_counter_card_ = 0;
  int ms_counter_low_ = 0;
  int ms_counter_high_ = 0;
  std::vector<uint32_t> ms_announce_codes_;
  size_t ms_announce_lo_ = 0;
  size_t ms_announce_hi_ = 0;
  std::vector<std::pair<size_t, size_t>> ms_announce_ranges_;
  uint8_t ms_place_player_ = 0;
  int n_places_ = 0;
  std::vector<ActionPlace> ms_places_;
  std::vector<ActionPlace> ms_selected_places_;

  // discard hand cards
  bool discard_hand_ = false;

  // replay
  bool record_ = false;
  FILE* fp_ = nullptr;
  bool is_recording = false;

  // MSG_SELECT_COUNTER
  int n_counters_ = 0;
  int n_sort_cards_ = 0;

  std::mt19937 gen_;

  std::mt19937 duel_gen_;

  static std::vector<uint8_t> read_binary_asset(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      throw std::runtime_error("Unable to open Structured-lite asset: " + path);
    }
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(input), {});
  }

  void windbot_close() {
#ifndef _WIN32
    if (windbot_fd_ >= 0) { ::close(windbot_fd_); windbot_fd_ = -1; }
    if (windbot_server_fd_ >= 0) { ::close(windbot_server_fd_); windbot_server_fd_ = -1; }
#endif
  }

  void windbot_listen() {
#ifdef _WIN32
    throw std::runtime_error("WindBot native bridge is currently supported on Linux only");
#else
    if (windbot_port_ <= 0)
      throw std::runtime_error("play_mode=windbot requires windbot_port > 0");
    windbot_server_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (windbot_server_fd_ < 0) throw std::runtime_error("WindBot socket() failed");
    int yes = 1;
    setsockopt(windbot_server_fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    timeval tv{windbot_timeout_, 0};
    setsockopt(windbot_server_fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(windbot_port_));
    if (inet_pton(AF_INET, windbot_host_.c_str(), &addr.sin_addr) != 1)
      throw std::runtime_error("windbot_host must be an IPv4 address");
    if (::bind(windbot_server_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0 ||
        ::listen(windbot_server_fd_, 1) < 0) {
      windbot_close();
      throw std::runtime_error("Unable to listen for WindBot at " + windbot_host_ + ":" + std::to_string(windbot_port_));
    }
#endif
  }

  static bool socket_read_all(int fd, byte* dst, size_t len) {
#ifndef _WIN32
    while (len) {
      auto n = ::recv(fd, dst, len, 0);
      if (n <= 0) return false;
      dst += n; len -= static_cast<size_t>(n);
    }
    return true;
#else
    return false;
#endif
  }

  void windbot_send(uint8_t proto, const byte* payload = nullptr, size_t len = 0) {
#ifndef _WIN32
    uint16_t packet_len = static_cast<uint16_t>(len + 1);
    byte header[3] = {static_cast<byte>(packet_len & 0xff),
                      static_cast<byte>((packet_len >> 8) & 0xff), proto};
    if (::send(windbot_fd_, header, 3, MSG_NOSIGNAL) != 3)
      throw std::runtime_error("WindBot disconnected while sending packet");
    size_t sent = 0;
    while (sent < len) {
      auto n = ::send(windbot_fd_, payload + sent, len - sent, MSG_NOSIGNAL);
      if (n <= 0) throw std::runtime_error("WindBot disconnected while sending payload");
      sent += static_cast<size_t>(n);
    }
#endif
  }

  std::pair<uint8_t, std::vector<byte>> windbot_recv() {
    byte header[3];
    if (!socket_read_all(windbot_fd_, header, 3))
      throw std::runtime_error("Timed out or disconnected waiting for WindBot");
    uint16_t len = static_cast<uint16_t>(header[0] | (header[1] << 8));
    if (len < 1 || len > 4096) throw std::runtime_error("Invalid WindBot packet length");
    std::vector<byte> payload(len - 1);
    if (!payload.empty() && !socket_read_all(windbot_fd_, payload.data(), payload.size()))
      throw std::runtime_error("Truncated WindBot packet");
    return {header[2], std::move(payload)};
  }

  void windbot_accept_lobby() {
#ifndef _WIN32
    if (windbot_fd_ >= 0) return;
    windbot_fd_ = ::accept(windbot_server_fd_, nullptr, nullptr);
    if (windbot_fd_ < 0) throw std::runtime_error("Timed out waiting for WindBot connection");
    timeval tv{windbot_timeout_, 0};
    setsockopt(windbot_fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    bool joined = false, ready = false;
    while (!ready) {
      auto packet = windbot_recv();
      if (packet.first == 0x12 && !joined) { // CTOS_JOIN_GAME
        const byte join_game[7] = {0, 0, 0, 0, 0, 0, 5};
        windbot_send(0x12, join_game, sizeof(join_game));
        byte type = static_cast<byte>((1 - ai_player_) | 0x10);
        windbot_send(0x13, &type, 1);
        joined = true;
      } else if (packet.first == 0x22) { // CTOS_HS_READY
        ready = true;
      }
    }
    windbot_send(0x15); // STOC_DUEL_START
    byte start[19]{};
    start[0] = 0x04; // MSG_START (network-only; not exposed by this core header)
    start[1] = static_cast<byte>(1 - ai_player_);
    start[2] = static_cast<byte>(rules_);
    auto put32 = [&](int off, uint32_t value) { std::memcpy(start + off, &value, 4); };
    auto put16 = [&](int off, uint16_t value) { std::memcpy(start + off, &value, 2); };
    put32(3, init_lp_); put32(7, init_lp_);
    put16(11, static_cast<uint16_t>(main_deck0_.size()));
    put16(13, static_cast<uint16_t>(extra_deck0_.size()));
    put16(15, static_cast<uint16_t>(main_deck1_.size()));
    put16(17, static_cast<uint16_t>(extra_deck1_.size()));
    windbot_send(0x01, start, sizeof(start));
    windbot_lobby_ready_ = true;
#endif
  }

  void windbot_game_message(const byte* payload, size_t len) {
    if (!windbot_lobby_ready_ || !len) return;
    if (payload[0] == MSG_DRAW && len >= 3 && payload[1] == ai_player_) {
      std::vector<byte> hidden(payload, payload + len);
      const size_t count = hidden[2];
      for (size_t i = 0; i < count && 3 + (i + 1) * 4 <= hidden.size(); ++i) {
        uint32_t code = 0;
        std::memcpy(&code, hidden.data() + 3 + i * 4, 4);
        code = (code & 0x80000000u) ? (code & 0x7fffffffu) : 0;
        std::memcpy(hidden.data() + 3 + i * 4, &code, 4);
      }
      windbot_send(0x01, hidden.data(), hidden.size());
      return;
    }
    windbot_send(0x01, payload, len);
  }

  static bool windbot_response_message(int msg) {
    switch (msg) {
      case MSG_RETRY:
      case MSG_SELECT_BATTLECMD:
      case MSG_SELECT_IDLECMD:
      case MSG_SELECT_EFFECTYN:
      case MSG_SELECT_YESNO:
      case MSG_SELECT_OPTION:
      case MSG_SELECT_CARD:
      case MSG_SELECT_UNSELECT_CARD:
      case MSG_SELECT_CHAIN:
      case MSG_SELECT_PLACE:
      case MSG_SELECT_POSITION:
      case MSG_SELECT_TRIBUTE:
      case MSG_SORT_CARD:
      case MSG_SELECT_COUNTER:
      case MSG_SELECT_SUM:
      case MSG_SELECT_DISFIELD:
      case MSG_ROCK_PAPER_SCISSORS:
      case MSG_ANNOUNCE_NUMBER:
      case MSG_ANNOUNCE_ATTRIB:
      case MSG_ANNOUNCE_CARD:
      case MSG_ANNOUNCE_RACE:
        return true;
      default:
        return false;
    }
  }

  void windbot_apply_response() {
    while (true) {
      auto packet = windbot_recv();
      if (packet.first == 0x01) { // CTOS_RESPONSE
        if (packet.second.empty() || packet.second.size() > sizeof(resp_buf_))
          throw std::runtime_error("Invalid WindBot response size");
        std::memset(resp_buf_, 0, sizeof(resp_buf_));
        std::memcpy(resp_buf_, packet.second.data(), packet.second.size());
        if (verbose_) {
          fmt::print("WindBot response for {} ({} bytes):", msg_to_string(msg_), packet.second.size());
          for (byte value : packet.second) fmt::print(" {:02x}", value);
          fmt::print("\n");
        }
        YGO_SetResponseb(pduel_, resp_buf_);
        ms_idx_ = -1;
        legal_actions_.clear();
        return;
      }
      if (packet.first == 0x15) windbot_send(0x15); // CTOS_TIME_CONFIRM
    }
  }


  static uint64_t sampler_splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
  }

  uint64_t sampler_random() {
    return sampler_splitmix64(sampler_seed_ + sampler_counter_++);
  }

  void initialize_deck_sampler(const std::string &manifest_path,
                               const std::string &seed_text,
                               const std::string &counters_text,
                               int env_id, uint64_t env_seed) {
    if (manifest_path.empty()) return;
    sampler_seed_ = std::stoull(seed_text) ^ sampler_splitmix64(env_seed);
    if (!counters_text.empty()) {
      std::istringstream counters(counters_text);
      std::string token;
      int index = 0;
      while (std::getline(counters, token, ',')) {
        if (index++ == env_id) {
          sampler_counter_ = std::stoull(token);
          break;
        }
      }
    }
    std::ifstream stream(manifest_path);
    if (!stream) {
      throw std::runtime_error("Cannot open deck sampling manifest: " + manifest_path);
    }
    std::string line;
    if (!std::getline(stream, line) || line != "deck-sampler-v1") {
      throw std::runtime_error("Unsupported deck sampling manifest");
    }
    while (std::getline(stream, line)) {
      if (line.empty()) continue;
      std::istringstream row(line);
      std::vector<std::string> fields;
      std::string field;
      while (std::getline(row, field, '\t')) fields.push_back(field);
      if (fields.size() == 2 && fields[0] == "reserve_ppm") {
        sampler_reserve_ppm_ = std::stoi(fields[1]);
      } else if (fields.size() == 2 && fields[0] == "anchor") {
        sampler_anchor_deck_ = fields[1];
      } else if (fields.size() == 7 && fields[0] == "deck") {
        int cluster_index = std::stoi(fields[1]);
        int family_index = std::stoi(fields[2]);
        int deck_index = std::stoi(fields[3]);
        if (cluster_index < 0 || family_index < 0 || deck_index < 0) {
          throw std::runtime_error("Negative deck sampler index");
        }
        while (sampler_clusters_.size() <= static_cast<size_t>(cluster_index)) {
          int index = sampler_clusters_.size();
          sampler_clusters_.push_back({"", index, {}});
        }
        auto &cluster = sampler_clusters_[cluster_index];
        if (cluster.cluster_id.empty()) cluster.cluster_id = fields[4];
        if (cluster.cluster_id != fields[4]) {
          throw std::runtime_error("Deck sampler cluster index collision");
        }
        while (cluster.families.size() <= static_cast<size_t>(family_index)) {
          int index = cluster.families.size();
          cluster.families.push_back({"", index, {}});
        }
        auto &family = cluster.families[family_index];
        if (family.family_id.empty()) family.family_id = fields[5];
        if (family.family_id != fields[5] ||
            family.decks.size() != static_cast<size_t>(deck_index)) {
          throw std::runtime_error("Deck sampler family/deck index collision");
        }
        family.decks.push_back({fields[6], deck_index});
      } else {
        throw std::runtime_error("Invalid deck sampling manifest row: " + line);
      }
    }
    if (sampler_reserve_ppm_ <= 0 || sampler_reserve_ppm_ >= 1000000 ||
        sampler_anchor_deck_.empty() || sampler_clusters_.empty()) {
      throw std::runtime_error("Incomplete deck sampling manifest");
    }
  }

  std::string sample_manifest_deck(PlayerId player) {
    SamplerSelection selection;
    if (sampler_random() % 1000000 < static_cast<uint64_t>(sampler_reserve_ppm_)) {
      selection.cluster_index = -1;
      selection.family_index = -1;
      selection.deck_index = -1;
      selection.is_anchor = 1;
      selection.counter = sampler_counter_;
      sampler_selection_[player] = selection;
      return sampler_anchor_deck_;
    }
    auto &cluster = sampler_clusters_[sampler_random() % sampler_clusters_.size()];
    auto &family = cluster.families[sampler_random() % cluster.families.size()];
    auto &deck = family.decks[sampler_random() % family.decks.size()];
    selection.cluster_index = cluster.cluster_index;
    selection.family_index = family.family_index;
    selection.deck_index = deck.deck_index;
    selection.counter = sampler_counter_;
    sampler_selection_[player] = selection;
    return deck.deck_id;
  }

public:
  // step return
  float ret_reward_ = 0;
  int ret_win_reason_ = 0;

  std::string timeout_diagnostic() const {
    const auto &seat0 = sampler_selection_[0];
    const auto &seat1 = sampler_selection_[1];
    return fmt::format(
        "env={} decks=[{},{}] clusters=[{},{}] families=[{},{}] "
        "deck_indices=[{},{}] anchors=[{},{}] sampler_counters=[{},{}] "
        "step={} turn={} phase={} lp=[{},{}] msg={}({}) engine_flag={} "
        "buffer={}/{} turn_player={}",
        env_id_, deck_name_[0], deck_name_[1],
        seat0.cluster_index, seat1.cluster_index,
        seat0.family_index, seat1.family_index,
        seat0.deck_index, seat1.deck_index,
        seat0.is_anchor, seat1.is_anchor,
        seat0.counter, seat1.counter,
        step_count_, turn_count_, current_phase_, lp_[0], lp_[1],
        msg_, msg_to_string(msg_), eng_flag_, dp_, dl_, tp_);
  }

  bool is_done() const { return done_; }

  YGOProEnvImpl();

  YGOProEnvImpl(const EnvSpec<YGOProEnvFns> &spec, uint64_t env_seed,
                int env_id = 0)
      : spec_(spec), env_id_(env_id), dist_int_(0, 0xffffffff),
        deck1_(spec.config["deck1"_]), deck2_(spec.config["deck2"_]),
        player_(spec.config["player"_]), players_{nullptr, nullptr},
        play_modes_(parse_play_modes(spec.config["play_mode"_])),
        verbose_(spec.config["verbose"_]), record_(spec.config["record"_]),
        n_history_actions_(spec.config["n_history_actions"_]),
        greedy_reward_(spec.config["greedy_reward"_]),
        windbot_host_(spec.config["windbot_host"_]),
        windbot_port_(spec.config["windbot_port"_]),
        windbot_timeout_(spec.config["windbot_timeout"_]) {
    // Replay recording is independent of human-readable decision logging.
    // Keeping verbose optional also avoids invoking incomplete debug formatters
    // for automatically resolved self-play actions.
    // fmt::println("env_id: {}, seed: {}, x: {}", env_id_, seed_, dist_int_(gen_));

    gen_ = std::mt19937(env_seed);
    duel_gen_ = std::mt19937(dist_int_(gen_));
    initialize_deck_sampler(spec.config["deck_sampling_manifest"_],
                            spec.config["deck_sampler_seed"_],
                            spec.config["deck_sampler_counters"_], env_id,
                            env_seed);

    int max_options = spec.config["max_options"_];
    int n_action_feats = spec.state_spec["obs:actions_"_].shape[1];
    history_actions_1_ = TArray<uint8_t>(Array(
        ShapeSpec(sizeof(uint8_t), {n_history_actions_, n_action_feats + 2})));
    history_actions_2_ = TArray<uint8_t>(Array(
        ShapeSpec(sizeof(uint8_t), {n_history_actions_, n_action_feats + 2})));
    if (structured_enabled()) {
      const std::string asset_dir = spec.config["semantic_asset_dir"_];
      if (!asset_dir.empty()) {
        card_semantics_table_ = read_binary_asset(asset_dir + "/card-semantics.u8");
        effect_tags_table_ = read_binary_asset(asset_dir + "/effect-tags.u8");
        effect_tag_confidence_table_ =
            read_binary_asset(asset_dir + "/effect-tag-confidence.u8");
        const size_t rows = card_ids_.size() + 1;
        if (card_semantics_table_.size() != rows * 32 ||
            effect_tags_table_.size() != rows * 16 ||
            effect_tag_confidence_table_.size() != rows * 16) {
          throw std::runtime_error("Structured-lite semantic table dimensions do not match code list");
        }
      }
    }
    if (std::find(play_modes_.begin(), play_modes_.end(), kWindBot) != play_modes_.end()) {
      windbot_listen();
    }
  }

  int max_options() const { return spec_.config["max_options"_]; }

  int max_cards() const { return spec_.config["max_cards"_]; }

  bool structured_enabled() const {
    return spec_.config["observation_schema"_] == "structured-lite-v1";
  }

  bool done() const { return done_; }

  bool random_mode() const { return play_modes_.size() > 1; }

  bool self_play() const {
    return std::find(play_modes_.begin(), play_modes_.end(), kSelfPlay) !=
           play_modes_.end();
  }

  void update_time_stat(const clock_t& start, uint64_t time_count, double& time_stat) {
    double seconds = static_cast<double>(clock() - start) / CLOCKS_PER_SEC;
    time_stat = time_stat * (static_cast<double>(time_count) /
      (time_count + 1)) + seconds / (time_count + 1);
  }

  // void update_time_stat(const std::string& deck, double seconds) {
  //   uint64_t& time_count = deck_time_count_[deck];
  //   double& time_stat = deck_time_[deck];
  //   time_stat = time_stat * (static_cast<double>(time_count) /
  //     (time_count + 1)) + seconds / (time_count + 1);
  //   time_count++;
  // }

  MDuel new_duel(uint32_t seed) {
    auto pduel = YGO_CreateDuel(seed);
    MDuel mduel{pduel, seed};

    for (PlayerId i = 0; i < 2; i++) {
      YGO_SetPlayerInfo(pduel, i, init_lp_, startcount_, drawcount_);
      auto [main_deck, extra_deck, deck_name] = load_deck(pduel, i, duel_gen_);
      if (i == 0) {
        mduel.main_deck0 = main_deck;
        mduel.extra_deck0 = extra_deck;
        mduel.deck_name0 = deck_name;
      } else {
        mduel.main_deck1 = main_deck;
        mduel.extra_deck1 = extra_deck;
        mduel.deck_name1 = deck_name;
      }
    }
    YGO_StartDuel(pduel, duel_options_);
    return mduel;
  }

  void reset() {
    // clock_t start = clock();

    if (random_mode()) {
      play_mode_ = play_modes_[dist_int_(gen_) % play_modes_.size()];
    } else {
      play_mode_ = play_modes_[0];
    }

    if (play_mode_ != kSelfPlay) {
      if (player_ == -1) {
        ai_player_ = dist_int_(gen_) % 2;
      } else {
        ai_player_ = player_;
      }
    }

    turn_count_ = 0;
    tp_ = 0;
    current_phase_ = 0;
    ms_idx_ = -1;
    n_places_ = 0;
    ms_places_.clear();
    ms_selected_places_.clear();
    chain_depth_ = 0;
    active_chain_source_ = VisibleCardRef{};
    public_events_.clear();
    public_event_overflow_ = 0;
    selection_forced_ = false;
    selection_finishable_ = false;
    selection_cancelable_ = false;
    policy_back_cancel_suppressed_ = false;
    last_action_overflow_ = false;

    history_actions_1_.Zero();
    history_actions_2_.Zero();
    ha_p_1_ = 0;
    ha_p_2_ = 0;

    // clock_t _start = clock();

    intptr_t old_duel = pduel_;
    if (duel_started_) {
      YGO_EndDuel(pduel_);
    }
    MDuel mduel;
    mduel = new_duel(dist_int_(gen_));

    auto duel_seed = mduel.seed;
    pduel_ = mduel.pduel;

    deck_name_[0] = mduel.deck_name0;
    deck_name_[1] = mduel.deck_name1;
    main_deck0_ = mduel.main_deck0;
    extra_deck0_ = mduel.extra_deck0;
    main_deck1_ = mduel.main_deck1;
    extra_deck1_ = mduel.extra_deck1;

    for (PlayerId i = 0; i < 2; i++) {
      std::string nickname = i == 0 ? "Alice" : "Bob";
      if (i == ai_player_) {
        nickname = "Agent";
      }
      nickname_[i] = nickname;
      if ((play_mode_ == kHuman) && (i != ai_player_)) {
        players_[i] = std::make_unique<HumanPlayer>(nickname, init_lp_, i, verbose_);
      } else if (play_mode_ == kRandomBot) {
        players_[i] = std::make_unique<RandomAI>(max_options(), dist_int_(gen_), nickname, init_lp_, i, verbose_);
      } else if (play_mode_ == kFirstActionBot) {
        players_[i] = std::make_unique<FirstActionAI>(nickname, init_lp_, i, verbose_);
      } else {
        players_[i] = std::make_unique<GreedyAI>(nickname, init_lp_, i, verbose_);
      }
      lp_[i] = players_[i]->init_lp_;
    }

    if (record_) {
      if (is_recording && fp_ != nullptr) {
        fclose(fp_);
      }
      auto time_str = time_now();
      // Use last 4 digits of seed as unique id
      auto seed_ = duel_seed % 10000;
      std::string fname;
      while (true) {
        fname = fmt::format("./replay/a{} {:04d}.yrp", time_str, seed_);
        // check existence
        if (std::filesystem::exists(fname)) {
          seed_ = (seed_ + 1) % 10000;
        } else {
          break;
        } 
      }
      fp_ = fopen(fname.c_str(), "wb");
      if (!fp_) {
        throw std::runtime_error("Failed to open file for replay: " + fname);
      }

      is_recording = true;

      ReplayHeader rh;
      rh.id = 0x31707279;
      rh.version = 0x00001360;
      rh.flag = REPLAY_UNIFORM;
      rh.seed = duel_seed;
      rh.start_time = (unsigned int)time(nullptr);
      fwrite(&rh, sizeof(rh), 1, fp_);

      for (PlayerId i = 0; i < 2; i++) {
        uint16_t name[20];
        memset(name, 0, 40);
        std::string name_str = fmt::format("{} {}", nickname_[i], deck_name_[i]);
        if (name_str.size() > 20) {
          // truncate
          name_str = name_str.substr(0, 20);
        }
        fmt::println("name: {}", name_str);
        str_to_uint16(name_str.c_str(), name);
        fwrite(name, 40, 1, fp_);
      }

      ReplayWriteInt32(init_lp_);
      ReplayWriteInt32(startcount_);
      ReplayWriteInt32(drawcount_);
      ReplayWriteInt32(duel_options_);

      for (PlayerId i = 0; i < 2; i++) {
        auto &main_deck = i == 0 ? main_deck0_ : main_deck1_;
        auto &extra_deck = i == 0 ? extra_deck0_ : extra_deck1_;
        ReplayWriteInt32(main_deck.size());
        for (auto code : main_deck) {
          ReplayWriteInt32(code);
        }
        ReplayWriteInt32(extra_deck.size());
        for (int j = int(extra_deck.size()) - 1; j >= 0; --j) {
          ReplayWriteInt32(extra_deck[j]);
        }
      }

    }

    duel_started_ = true;
    eng_flag_ = 0;
    winner_ = 255;
    win_reason_ = 255;
    discard_hand_ = false;

    done_ = false;
    step_count_ = 0;
    invalid_game_ = false;
    termination_reason_ = kTerminationNone;

    if (play_mode_ == kWindBot) {
      windbot_accept_lobby();
    }

    // update_time_stat(_start, reset_time_count_, reset_time_2_);
    // _start = clock();

    next();

    ret_reward_ = 0;
    ret_win_reason_ = 0;
  }

  void init_multi_select(
    int min, int max, int must, const std::vector<std::string> &specs,
    int mode = 0, const std::vector<std::vector<int>> &combs = {}) {
    ms_idx_ = 0;
    ms_mode_ = mode;
    ms_min_ = min;
    ms_max_ = max;
    ms_must_ = must;
    ms_specs_ = specs;
    ms_r_idxs_.clear();
    ms_spec2idx_.clear();

    for (int j = 0; j < ms_specs_.size(); ++j) {
      const auto &spec = ms_specs_[j];
      ms_spec2idx_[spec] = j;
    }

    if (ms_mode_ == 0) {
      for (int j = 0; j < ms_specs_.size(); ++j) {
        const auto &spec = ms_specs_[j];
        legal_actions_.push_back(LegalAction::from_spec(spec));
      }
      if (ms_min_ == 0) {
        legal_actions_.push_back(LegalAction::finish());
      }
      if (selection_cancelable_) {
        legal_actions_.push_back(LegalAction::cancel());
      }
    } else if (ms_mode_ == 3) {
      prepare_weighted_selection();
    } else if (ms_mode_ == 4) {
      prepare_sort_selection();
    } else {
      ms_combs_ = combs;
      selection_finishable_ = std::any_of(
          ms_combs_.begin(), ms_combs_.end(),
          [](const std::vector<int> &combination) {
            return combination.empty();
          });
      _callback_multi_select_2_prepare();
    }
  }

  std::array<uint8_t, 3> encode_place(ActionPlace place,
                                      uint8_t player) const {
    return core_encode_place(place, player);
  }

  void finish_place_selection() {
    if (ms_selected_places_.empty()) {
      resp_buf_[0] = 0;
      resp_buf_[1] = 0;
      resp_buf_[2] = 0;
    } else {
      for (int index = 0; index < static_cast<int>(ms_selected_places_.size());
           ++index) {
        const auto encoded = encode_place(ms_selected_places_[index],
                                          ms_place_player_);
        resp_buf_[index * 3] = encoded[0];
        resp_buf_[index * 3 + 1] = encoded[1];
        resp_buf_[index * 3 + 2] = encoded[2];
      }
    }
    ms_idx_ = -1;
    YGO_SetResponseb(pduel_, resp_buf_);
  }

  void callback_place_select(int action_index) {
    const auto action = legal_actions_[action_index];
    if (action.finish_) {
      if (!selection_finishable_ || !ms_selected_places_.empty()) {
        throw std::runtime_error("Invalid empty place selection");
      }
      finish_place_selection();
      return;
    }
    auto selected = std::find(ms_places_.begin(), ms_places_.end(),
                              action.place_);
    if (selected == ms_places_.end()) {
      throw std::runtime_error("Selected place is not available");
    }
    ms_selected_places_.push_back(*selected);
    ms_places_.erase(selected);
    ++ms_idx_;
    if (static_cast<int>(ms_selected_places_.size()) == n_places_) {
      finish_place_selection();
    }
  }

  void prepare_announce_card_selection() {
    legal_actions_.clear();
    ms_announce_ranges_.clear();
    const size_t remaining = ms_announce_hi_ - ms_announce_lo_;
    if (remaining == 0) {
      throw std::runtime_error("announce card has no remaining candidates");
    }
    if (remaining <= static_cast<size_t>(max_options())) {
      for (size_t i = ms_announce_lo_; i < ms_announce_hi_; ++i) {
        const uint32_t code = ms_announce_codes_[i];
        LegalAction action;
        action.cid_ = c_get_card_id(code);
        action.response_ = code;
        legal_actions_.push_back(action);
      }
      callback_ = [this](int index) {
        const uint32_t code = legal_actions_[index].response_;
        ms_idx_ = -1;
        YGO_SetResponsei(pduel_, code);
      };
      return;
    }
    const size_t branches = std::min<size_t>(16, max_options());
    if (branches < 2) {
      throw std::runtime_error("announce card requires at least two action slots");
    }
    const size_t chunk = (remaining + branches - 1) / branches;
    for (size_t lo = ms_announce_lo_; lo < ms_announce_hi_; lo += chunk) {
      const size_t hi = std::min(ms_announce_hi_, lo + chunk);
      ms_announce_ranges_.emplace_back(lo, hi);
      LegalAction action;
      action.cid_ = c_get_card_id(ms_announce_codes_[lo]);
      action.number_ = static_cast<uint32_t>(ms_announce_ranges_.size() - 1);
      legal_actions_.push_back(action);
    }
    callback_ = [this](int index) {
      const auto [lo, hi] = ms_announce_ranges_[index];
      ms_announce_lo_ = lo;
      ms_announce_hi_ = hi;
      ++ms_idx_;
    };
  }

  void handle_multi_select() {
    legal_actions_.clear();
    if (ms_mode_ == 0) {
      for (int j = 0; j < ms_specs_.size(); ++j) {
        if (ms_spec2idx_.find(ms_specs_[j]) != ms_spec2idx_.end()) {
          legal_actions_.push_back(
            LegalAction::from_spec(ms_specs_[j]));
        }
      }
      if (ms_idx_ == ms_max_ - 1) {
        if (ms_idx_ >= ms_min_) {
          legal_actions_.push_back(LegalAction::finish());
        }
        callback_ = [this](int idx) {
          _callback_multi_select(idx, true);
        };
      } else if (ms_idx_ >= ms_min_) {
        legal_actions_.push_back(LegalAction::finish());
        callback_ = [this](int idx) {
          _callback_multi_select(idx, false);
        };
      } else {
        callback_ = [this](int idx) {
          _callback_multi_select(idx, false);
        };    
      }
      if (selection_cancelable_) {
        legal_actions_.push_back(LegalAction::cancel());
      }
    } else if (ms_mode_ == 1) {
      _callback_multi_select_2_prepare();
      callback_ = [this](int idx) {
        _callback_multi_select_2(idx);
      };
    } else if (ms_mode_ == 3) {
      prepare_weighted_selection();
      callback_ = [this](int idx) { callback_weighted_selection(idx); };
    } else if (ms_mode_ == 4) {
      prepare_sort_selection();
      callback_ = [this](int idx) { callback_sort_selection(idx); };
    } else if (ms_mode_ == 5) {
      prepare_counter_selection();
      callback_ = [this](int idx) { callback_counter_selection(idx); };
    } else if (ms_mode_ == 6) {
      prepare_announce_card_selection();
    } else if (ms_mode_ == 2) {
      for (ActionPlace place : ms_places_) {
        legal_actions_.push_back(LegalAction::place(place));
      }
      if (selection_finishable_ && ms_selected_places_.empty()) {
        legal_actions_.push_back(LegalAction::finish());
      }
      callback_ = [this](int idx) { callback_place_select(idx); };
    } else {
      throw std::runtime_error("Invalid multi-select mode");
    }
  }

  int get_ms_spec_idx(const std::string &spec) const {
    auto it = ms_spec2idx_.find(spec);
    if (it != ms_spec2idx_.end()) {
      return it->second;
    }
    // A missing spec means the protocol-derived selection state and the
    // action exposed to the model have diverged. Never guess an index here:
    // that would turn a protocol error into a plausible but wrong response.
    show_deck(0);
    show_deck(1);
    show_buffer();
    show_turn();
    fmt::println("MS: idx: {}, mode: {}, min: {}, max: {}, must: {}, specs: {}, combs: {}, r_idx: {}", ms_idx_, ms_mode_, ms_min_, ms_max_, ms_must_, ms_specs_, ms_combs_, ms_r_idxs_);
    fmt::print("ms_spec2idx: ");
    for (const auto &[k, v] : ms_spec2idx_) {
      fmt::print("({}, {}), ", k, v);
    }
    fmt::print("\n");
    throw std::runtime_error(fmt::format(
        "[protocol boundary] multi-select spec not found: {}", spec));
  }

  void _callback_multi_select_2(int idx) {
    const auto &action = legal_actions_[idx];
    if (action.finish_) {
      const bool can_finish = std::any_of(
          ms_combs_.begin(), ms_combs_.end(),
          [](const std::vector<int> &combination) { return combination.empty(); });
      if (!can_finish) throw std::runtime_error("Invalid weighted selection finish");
      _callback_multi_select_2_finish();
      return;
    }
    idx = get_ms_spec_idx(action.spec_);
    if (idx == -1) {
      // TODO(2): find the root cause
      std::vector<std::string> specs;
      for (const auto &la : legal_actions_) {
        specs.push_back(la.spec_);
      }
      fmt::println("specs: {}, idx: {}, spec: {}", specs, idx, action.spec_);
      throw std::runtime_error("Spec not found");
    }
    ms_r_idxs_.push_back(idx);
    std::vector<std::vector<int>> combs;
    for (auto &c : ms_combs_) {
      if (!c.empty() && c[0] == idx) {
        c.erase(c.begin());
        if (c.empty()) {
          // TODO: maybe finish too early
          _callback_multi_select_2_finish();
          return;
        } else {
          combs.push_back(c);
        }
      }
    }
    ms_idx_++;
    ms_combs_ = combs;
  }

  void _callback_multi_select_2_prepare() {
    std::set<int> comb;
    bool can_finish = false;
    for (const auto &c : ms_combs_) {
      if (c.empty()) can_finish = true;
      else comb.insert(c[0]);
    }
    for (auto &i : comb) {
      const auto &spec = ms_specs_[i];
      legal_actions_.push_back(LegalAction::from_spec(spec));
    }
    if (can_finish) legal_actions_.push_back(LegalAction::finish());
  }

  void _callback_multi_select_2_finish() {
    ms_idx_ = -1;
    resp_buf_[0] = ms_r_idxs_.size() + ms_must_;
    for (int i = 0; i < ms_must_; ++i) {
      resp_buf_[i + 1] = 0;
    }
    for (int i = 0; i < ms_r_idxs_.size(); ++i) {
      resp_buf_[i + ms_must_ + 1] = ms_r_idxs_[i];
    }
    YGO_SetResponseb(pduel_, resp_buf_);
  }

  void prepare_weighted_selection() {
    legal_actions_.clear();
    auto [choices, finishable] = core_weighted_staged_choices(
        ms_weighted_kind_, ms_weighted_must_, ms_weighted_optional_,
        ms_r_idxs_, ms_weighted_target_, ms_min_, ms_max_);
    if (choices.empty() && !finishable) {
      throw std::runtime_error(
          "[protocol boundary] weighted selection has no core-valid completion");
    }
    selection_finishable_ = finishable;
    for (int index : choices) {
      legal_actions_.push_back(LegalAction::from_spec(ms_specs_[index]));
    }
    if (finishable) legal_actions_.push_back(LegalAction::finish());
    if (selection_cancelable_) legal_actions_.push_back(LegalAction::cancel());
  }

  void callback_weighted_selection(int action_index) {
    const auto &action = legal_actions_[action_index];
    if (action.act_ == ActionAct::Cancel) {
      if (!selection_cancelable_) {
        throw std::runtime_error("Invalid weighted selection cancel");
      }
      ms_idx_ = -1;
      YGO_SetResponsei(pduel_, -1);
      return;
    }
    if (action.finish_) {
      if (!selection_finishable_) {
        throw std::runtime_error("Invalid weighted selection finish");
      }
      _callback_multi_select_2_finish();
      return;
    }
    const int index = get_ms_spec_idx(action.spec_);
    if (!ms_r_idxs_.empty() && index <= ms_r_idxs_.back()) {
      throw std::runtime_error("[protocol boundary] nonmonotonic weighted choice");
    }
    ms_r_idxs_.push_back(index);
    ++ms_idx_;
  }

  void prepare_sort_selection() {
    legal_actions_.clear();
    for (int index = 0; index < static_cast<int>(ms_specs_.size()); ++index) {
      if (std::find(ms_r_idxs_.begin(), ms_r_idxs_.end(), index) ==
          ms_r_idxs_.end()) {
        legal_actions_.push_back(LegalAction::from_spec(ms_specs_[index]));
      }
    }
    selection_finishable_ = ms_r_idxs_.empty();
    if (selection_finishable_) legal_actions_.push_back(LegalAction::finish());
  }

  void callback_sort_selection(int action_index) {
    const auto &action = legal_actions_[action_index];
    if (action.finish_) {
      if (!ms_r_idxs_.empty()) {
        throw std::runtime_error("Invalid partial sort finish");
      }
      resp_buf_[0] = 0xff;
      ms_idx_ = -1;
      YGO_SetResponseb(pduel_, resp_buf_);
      return;
    }
    const int index = get_ms_spec_idx(action.spec_);
    if (std::find(ms_r_idxs_.begin(), ms_r_idxs_.end(), index) !=
        ms_r_idxs_.end()) {
      throw std::runtime_error("Duplicate sort index");
    }
    ms_r_idxs_.push_back(index);
    ++ms_idx_;
    if (ms_r_idxs_.size() == ms_specs_.size()) {
      for (int offset = 0; offset < static_cast<int>(ms_r_idxs_.size());
           ++offset) {
        resp_buf_[offset] = static_cast<uint8_t>(ms_r_idxs_[offset]);
      }
      ms_idx_ = -1;
      YGO_SetResponseb(pduel_, resp_buf_);
    }
  }

  void prepare_counter_selection() {
    legal_actions_.clear();
    while (ms_counter_card_ < static_cast<int>(ms_counter_capacities_.size())) {
      int future = 0;
      for (int index = ms_counter_card_ + 1;
           index < static_cast<int>(ms_counter_capacities_.size()); ++index) {
        future += ms_counter_capacities_[index];
      }
      const int feasible_low = std::max(0, ms_counter_remaining_ - future);
      const int feasible_high = std::min(
          ms_counter_remaining_, ms_counter_capacities_[ms_counter_card_]);
      if (feasible_low > feasible_high) {
        throw std::runtime_error(
            "[protocol boundary] counter allocation has no valid completion");
      }
      if (ms_counter_low_ < feasible_low || ms_counter_high_ > feasible_high ||
          ms_counter_low_ > ms_counter_high_) {
        ms_counter_low_ = feasible_low;
        ms_counter_high_ = feasible_high;
      }
      if (ms_counter_low_ != ms_counter_high_) break;
      const auto amount = static_cast<uint16_t>(ms_counter_low_);
      ms_counter_allocations_.push_back(amount);
      ms_counter_remaining_ -= amount;
      ++ms_counter_card_;
      ms_counter_low_ = 0;
      ms_counter_high_ = 0xffff;
    }
    if (ms_counter_card_ == static_cast<int>(ms_counter_capacities_.size())) {
      if (ms_counter_remaining_ != 0 ||
          ms_counter_allocations_.size() * 2 > sizeof(resp_buf_)) {
        throw std::runtime_error(
            "[protocol boundary] invalid counter allocation response size");
      }
      for (int index = 0; index < static_cast<int>(ms_counter_allocations_.size());
           ++index) {
        const uint16_t amount = ms_counter_allocations_[index];
        resp_buf_[index * 2] = static_cast<uint8_t>(amount);
        resp_buf_[index * 2 + 1] = static_cast<uint8_t>(amount >> 8);
      }
      ms_idx_ = -1;
      YGO_SetResponseb(pduel_, resp_buf_);
      return;
    }
    legal_actions_.push_back(LegalAction::number(0));
    legal_actions_.push_back(LegalAction::number(1));
  }

  void callback_counter_selection(int action_index) {
    if (action_index < 0 || action_index > 1 ||
        ms_counter_low_ >= ms_counter_high_) {
      throw std::runtime_error("Invalid counter allocation choice");
    }
    const int middle = ms_counter_low_ +
                       (ms_counter_high_ - ms_counter_low_) / 2;
    if (action_index == 0) ms_counter_high_ = middle;
    else ms_counter_low_ = middle + 1;
    ++ms_idx_;
  }

  void _callback_multi_select(int idx, bool finish) {
    const auto &action = legal_actions_[idx];
    if (action.act_ == ActionAct::Cancel) {
      if (!selection_cancelable_) {
        throw std::runtime_error("Invalid multi-select cancel");
      }
      ms_idx_ = -1;
      YGO_SetResponsei(pduel_, -1);
      return;
    }
    // fmt::println("Select card: {}, finish: {}", option, finish);
    if (action.finish_) {
      finish = true;
    } else {
      idx = get_ms_spec_idx(action.spec_);
      if (idx != -1) {
        ms_r_idxs_.push_back(idx);
      } else {
        // TODO(2): find the root cause
        std::vector<std::string> specs;
        for (const auto &la : legal_actions_) {
          specs.push_back(la.spec_);
        }
        fmt::println("specs: {}, idx: {}, spec: {}", specs, idx, action.spec_);
        ms_idx_ = -1;
        resp_buf_[0] = ms_min_;
        for (int i = 0; i < ms_min_; ++i) {
          resp_buf_[i + 1] = i;
        }
        YGO_SetResponseb(pduel_, resp_buf_);
        return;
      }
    }
    if (finish) {
      ms_idx_ = -1;
      resp_buf_[0] = ms_r_idxs_.size();
      for (int i = 0; i < ms_r_idxs_.size(); ++i) {
        resp_buf_[i + 1] = ms_r_idxs_[i];
      }
      YGO_SetResponseb(pduel_, resp_buf_);
    } else {
      ms_idx_++;
      ms_spec2idx_.erase(action.spec_);
    }
  }

  void update_history_actions(PlayerId player, const LegalAction& action) {
    auto& ha_p = player == 0 ? ha_p_1_ : ha_p_2_;
    auto& history_actions = player == 0 ? history_actions_1_ : history_actions_2_;
    ha_p--;
    if (ha_p < 0) {
      ha_p = n_history_actions_ - 1;
    }
    history_actions[ha_p].Zero();
    _set_obs_action(history_actions, ha_p, action);
    // Spec index not available in history actions
    history_actions[ha_p](0) = 0;
    // history_actions[ha_p](12) = static_cast<uint8_t>(player);
    history_actions[ha_p](12) = static_cast<uint8_t>(turn_count_);
    history_actions[ha_p](13) = static_cast<uint8_t>(phase_to_id(current_phase_));
  }

  void filter_policy_navigational_back() {
    policy_back_cancel_suppressed_ = false;
    if (play_mode_ == kHuman) return;
    if (msg_ != MSG_SELECT_CARD && msg_ != MSG_SELECT_TRIBUTE &&
        msg_ != MSG_SELECT_UNSELECT_CARD) {
      return;
    }
    const auto is_cancel = [](const LegalAction &action) {
      return action.act_ == ActionAct::Cancel;
    };
    const bool has_cancel = std::any_of(
        legal_actions_.begin(), legal_actions_.end(), is_cancel);
    const bool has_progress = std::any_of(
        legal_actions_.begin(), legal_actions_.end(),
        [&is_cancel](const LegalAction &action) { return !is_cancel(action); });
    if (!has_cancel || !has_progress) return;
    legal_actions_.erase(
        std::remove_if(legal_actions_.begin(), legal_actions_.end(), is_cancel),
        legal_actions_.end());
    policy_back_cancel_suppressed_ = true;
  }

  void show_deck(const std::vector<CardCode> &deck, const std::string &prefix) const {
    fmt::print("{} deck: [", prefix);
    for (int i = 0; i < deck.size(); i++) {
      fmt::print(" '{}'", c_get_card(deck[i]).name());
    }
    fmt::print(" ]\n");
  }

  void show_turn() const {
    fmt::println("turn: {}, phase: {}, tplayer: {}", turn_count_, phase_to_string(current_phase_), tp_);
  }

  void show_buffer() const {
    fmt::println("msg: {}, dp: {}, dl: {}", msg_to_string(msg_), dp_, dl_);
    for (int i = 0; i < dl_; ++i) {
      fmt::print("{:02x} ", data_[i]);
    }
    fmt::print("\n");
  }

  void show_deck(PlayerId player) const {
    fmt::print("Player {}'s deck: {}\n", player, deck_name_[player]);
    // show_deck(player == 0 ? main_deck0_ : main_deck1_, "Main");
    // show_deck(player == 0 ? extra_deck0_ : extra_deck1_, "Extra");
  }

  void show_history_actions(PlayerId player) const {
    const auto &ha = player == 0 ? history_actions_1_ : history_actions_2_;
    // print card ids of history actions
    for (int i = 0; i < n_history_actions_; ++i) {
      fmt::print("history {}\n", i);
      uint8_t msg_id = uint8_t(ha(i, 3));
      int msg = _msgs[msg_id - 1];
      fmt::print("msg: {},", msg_to_string(msg));
      uint8_t v1 = ha(i, 1);
      uint8_t v2 = ha(i, 2);
      CardId card_id = (static_cast<CardId>(v1) << 8) + static_cast<CardId>(v2);
      fmt::print(" {};", card_id);
      for (int j = 4; j < ha.Shape()[1]; j++) {
        fmt::print(" {}", uint8_t(ha(i, j)));
      }
      fmt::print("\n");
    }
  }

  void step(int idx) {
    if (!callback_) {
      throw std::runtime_error(fmt::format(
          "Missing Step callback for message {} ({}) with {} legal actions",
          msg_, msg_to_string(msg_), legal_actions_.size()));
    }
    if (idx < 0 || idx >= static_cast<int>(legal_actions_.size())) {
      throw std::runtime_error(fmt::format(
          "[protocol boundary] action index {} out of range [0, {}) for "
          "message {} ({})",
          idx, legal_actions_.size(), msg_, msg_to_string(msg_)));
    }
    callback_(idx);
    update_history_actions(to_play_, legal_actions_[idx]);

    PlayerId player = to_play_;

    if (verbose_) {
      show_decision(idx);
    }

    if (ms_idx_ != -1) {
      handle_multi_select();
      filter_policy_navigational_back();
      if (ms_idx_ == -1 && legal_actions_.empty()) {
        next();
      }
    } else {
      next();
    }

    step_count_++;
    if (!done_ && (step_count_ >= spec_.config["max_steps"_])) {
      invalid_game_ = true;
      termination_reason_ = kTerminationMaxSteps;
      fmt::println("Env max steps: {}", timeout_diagnostic());
      done_ = true;
      legal_actions_.clear();
      if (duel_started_) {
        YGO_EndDuel(pduel_);
        duel_started_ = false;
      }
    }

    float reward = 0;
    int reason = 0;
    if (done_) {
      if (!invalid_game_) {
        float base_reward;
        if (greedy_reward_) {
          if (winner_ == 0) {
            if (turn_count_ <= 1) {
              // FTK
              base_reward = 16.0;
            } else if (turn_count_ <= 3) {
              base_reward = 8.0;
            } else if (turn_count_ <= 5) {
              base_reward = 4.0;
            } else if (turn_count_ <= 7) {
              base_reward = 2.0;
            } else {
              base_reward = 0.5 + 1.0 / (turn_count_ - 7);
            }
          } else {
            if (turn_count_ <= 1) {
              base_reward = 8.0;
            } else if (turn_count_ <= 3) {
              base_reward = 4.0;
            } else if (turn_count_ <= 5) {
              base_reward = 2.0;
            } else {
              base_reward = 0.5 + 1.0 / (turn_count_ - 5);
            }
          }
        } else {
          base_reward = 1.0;
        }

        if (play_mode_ == kSelfPlay) {
          // if (spec_.config["oppo_info"_]) {
          if (false) {
            reward = winner_ == 0 ? base_reward : -base_reward;
          } else {
            // to_play_ is the previous player
            reward = winner_ == player ? base_reward : -base_reward;
          }
        } else {
          reward = winner_ == ai_player_ ? base_reward : -base_reward;
        }

        if (win_reason_ == 0x01) {
          reason = 1;
        } else if (win_reason_ == 0x02) {
          reason = -1;
        }
      }

      if (record_) {
        if (!is_recording || fp_ == nullptr) {
          throw std::runtime_error("Recording is not started");
        }
        fclose(fp_);
        is_recording = false;
      }
    }


    // update_time_stat(start, step_time_count_, step_time_);
    // step_time_count_++;

    // double step_time = 0;
    // if (done_) {
    //   step_time = step_time_;
    //   step_time_ = 0;
    //   step_time_count_ = 0;
    // }

    // if (done_) {
    //   update_time_stat(deck_name_[0], step_time_);
    //   update_time_stat(deck_name_[1], step_time_);
    //   step_time_ = 0;
    //   step_time_count_ = 0;
    // }
    // if (step_time_count_ % 3000 == 0) {
    //   fmt::println("Step time: {:.3f}", step_time_ * 1000);
    // }
    ret_reward_ = reward;
    ret_win_reason_ = reason;
  }

  using YGOProEnvSpec = EnvSpec<YGOProEnvFns>;
  using State =
      Dict<typename YGOProEnvSpec::StateKeys,
           typename SpecToTArray<typename YGOProEnvSpec::StateSpec::Values>::Type>;

  void WriteState(State &state) {
    float reward = ret_reward_;
    int win_reason = ret_win_reason_;
    int n_options = legal_actions_.size();
    state["reward"_] = reward;
    state["info:to_play"_] = int(to_play_);
    state["info:is_selfplay"_] = int(play_mode_ == kSelfPlay);
    state["info:win_reason"_] = win_reason;
    state["info:invalid_game"_] = int(invalid_game_);
    state["info:termination_reason"_] = int(termination_reason_);
    state["info:episode_steps"_] = int(step_count_);
    state["info:turn_count"_] = std::min(turn_count_, spec_.config["max_steps"_]);
    clear_legacy_state(state);
    if (structured_enabled()) {
      clear_structured_state(state);
    }
    for (int seat = 0; seat < 2; ++seat) {
      state["info:deck"_][seat] = deck_name_[seat].empty() ? -1 : deck_names_ids_[deck_name_[seat]];
      state["info:deck_cluster"_][seat] = sampler_selection_[seat].cluster_index;
      state["info:deck_family"_][seat] = sampler_selection_[seat].family_index;
      state["info:deck_member"_][seat] = sampler_selection_[seat].deck_index;
      state["info:deck_is_anchor"_][seat] = sampler_selection_[seat].is_anchor;
      state["info:deck_sampler_counter"_][seat] = sampler_selection_[seat].counter;
    }
    if (reward != 0.0) {
      state["info:step_time"_][0] = 0;
      state["info:step_time"_][1] = 0;
    }

    if (n_options == 0) {
      state["info:num_options"_] = 1;
      state["obs:global_"_][22] = uint8_t(1);
      // if (step_count_ >= spec_.config["max_steps"_]) {
      //   fmt::println("Max steps reached return");
      // }
      return;
    }

    SpecInfos spec_infos;
    std::vector<int> loc_n_cards;

    if (spec_.config["oppo_info"_]) {
      _set_obs_g_cards(state["obs:cards_"_], to_play_);
      auto [spec_infos_, loc_n_cards_] = _set_obs_mask(state["obs:mask_"_], to_play_);
      spec_infos = spec_infos_;
      loc_n_cards = loc_n_cards_;
    } else {
      auto [spec_infos_, loc_n_cards_] = _set_obs_cards(state["obs:cards_"_], to_play_);
      spec_infos = spec_infos_;
      loc_n_cards = loc_n_cards_;
    }

    _set_obs_global(state["obs:global_"_], to_play_, loc_n_cards);

    // we can't shuffle because idx must be stable in callback
    last_action_overflow_ = n_options > max_options();
    core_validate_action_capacity(n_options, max_options());

    state["info:num_options"_] = n_options;

    for (int i = 0; i < n_options; ++i) {
      auto &action = legal_actions_[i];
      action.msg_ = msg_;
      const auto &spec = action.spec_;
      if (!spec.empty()) {
        const auto& spec_info = find_spec_info(spec_infos, spec);
        action.spec_index_ = spec_info.index;
        if (action.cid_ == 0) {
          action.cid_ = spec_info.cid;
        }
      }
    }

    _set_obs_actions(state["obs:actions_"_], legal_actions_);
    if (structured_enabled()) {
      _set_obs_structured(state, spec_infos);
    }

    // write history actions

    auto ha_p = to_play_ == 0 ? ha_p_1_ : ha_p_2_;
    auto &history_actions = to_play_ == 0 ? history_actions_1_ : history_actions_2_;

    int offset = n_history_actions_ - ha_p;
    int n_h_action_feats = history_actions.Shape()[1];

    state["obs:h_actions_"_].Assign(
      (uint8_t *)history_actions[ha_p].Data(), n_h_action_feats * offset);
    state["obs:h_actions_"_][offset].Assign(
      (uint8_t *)history_actions.Data(), n_h_action_feats * ha_p);
    
    for (int i = 0; i < n_history_actions_; ++i) {
      if (uint8_t(state["obs:h_actions_"_](i, 3)) == 0) {
        break;
      }
      // state["obs:h_actions_"_](i, 12) = static_cast<uint8_t>(uint8_t(state["obs:h_actions_"_](i, 12)) == to_play_);
      int turn_diff = std::min(16, turn_count_ - uint8_t(state["obs:h_actions_"_](i, 12)));
      state["obs:h_actions_"_](i, 12) = static_cast<uint8_t>(turn_diff);
    }
    validate_model_domains(state);
  }

private:
  using SpecInfos = ankerl::unordered_dense::map<std::string, SpecInfo>;

  void clear_legacy_state(State &state) {
    // EnvPool reuses output buffers. Every padding row must be explicitly
    // zeroed or the policy mask can mistake a previous step's action for a
    // currently legal one, while stale card rows silently contaminate state.
    std::memset(state["obs:cards_"_].Data(), 0,
                max_cards() * 2 * 41 * sizeof(uint8_t));
    std::memset(state["obs:global_"_].Data(), 0, 23 * sizeof(uint8_t));
    std::memset(state["obs:actions_"_].Data(), 0,
                max_options() * 12 * sizeof(uint8_t));
    std::memset(state["obs:mask_"_].Data(), 0,
                max_cards() * 2 * 14 * sizeof(uint8_t));
  }

  void clear_structured_state(State &state) {
    std::memset(state["obs:visible_card_ids_"_].Data(), 0,
                max_cards() * 2 * 2 * sizeof(uint8_t));
    std::memset(state["obs:card_semantics_"_].Data(), 0,
                max_cards() * 2 * 32 * sizeof(uint8_t));
    std::memset(state["obs:effect_tags_"_].Data(), 0,
                max_cards() * 2 * 16 * sizeof(uint8_t));
    std::memset(state["obs:effect_tag_confidence_"_].Data(), 0,
                max_cards() * 2 * 16 * sizeof(uint8_t));
    std::memset(state["obs:selection_"_].Data(), 0, 16 * sizeof(uint8_t));
    std::memset(state["obs:action_features_"_].Data(), 0,
                max_options() * 16 * sizeof(uint8_t));
    std::memset(state["obs:action_single_refs_"_].Data(), 0,
                max_options() * 4 * 3 * sizeof(uint16_t));
    const int ngr = spec_.config["max_group_references"_];
    std::memset(state["obs:action_group_refs_"_].Data(), 0,
                max_options() * 4 * ngr * 2 * sizeof(uint16_t));
    std::memset(state["obs:action_group_mask_"_].Data(), 0,
                max_options() * 4 * ngr * sizeof(uint8_t));
    const int npe = spec_.config["n_public_events"_];
    std::memset(state["obs:public_events_"_].Data(), 0,
                npe * 12 * sizeof(uint8_t));
    std::memset(state["obs:public_event_refs_"_].Data(), 0,
                npe * 4 * 3 * sizeof(uint16_t));
    std::memset(state["obs:structured_diagnostics_"_].Data(), 0,
                8 * sizeof(uint8_t));
  }

  [[noreturn]] void throw_domain_error(const std::string &tensor, int row,
                                       int field, uint32_t value,
                                       uint32_t upper_exclusive) const {
    std::ostringstream payload;
    for (int i = 0; i < dl_; ++i) {
      if (i) payload << ' ';
      payload << fmt::format("{:02x}", data_[i]);
    }
    throw std::runtime_error(fmt::format(
        "Protocol/model domain violation: message={}({}), tensor={}, row={}, "
        "field={}, value={}, expected=0..{}, dp={}, dl={}, payload=[{}]",
        msg_, msg_to_string(msg_), tensor, row, field, value,
        upper_exclusive - 1, dp_, dl_, payload.str()));
  }

  void validate_model_domains(State &state) const {
    auto check = [this](const std::string &tensor, int row, int field,
                        uint32_t value, uint32_t upper_exclusive) {
      if (value >= upper_exclusive) {
        throw_domain_error(tensor, row, field, value, upper_exclusive);
      }
    };
    auto check_card_id = [&](const std::string &tensor, int row,
                             uint32_t high, uint32_t low) {
      const uint32_t card_id = high * 256 + low;
      if (card_id > card_ids_.size()) {
        throw_domain_error(tensor, row, 1, card_id,
                           static_cast<uint32_t>(card_ids_.size() + 1));
      }
    };

    auto &cards = state["obs:cards_"_];
    const uint32_t card_upper[] = {9, 76, 2, 9, 2, 8, 27, 14, 16, 3};
    for (int row = 0; row < max_cards() * 2; ++row) {
      check_card_id("cards_", row, cards(row, 0), cards(row, 1));
      for (int field = 0; field < 10; ++field) {
        check("cards_", row, field + 2, cards(row, field + 2),
              card_upper[field]);
      }
      for (int field = 16; field < 41; ++field) {
        check("cards_", row, field, cards(row, field), 2);
      }
    }

    auto &global = state["obs:global_"_];
    check("global_", 0, 4, global(4), 20);
    check("global_", 0, 5, global(5), 11);
    check("global_", 0, 6, global(6), 2);
    check("global_", 0, 7, global(7), 2);
    for (int field = 8; field < 22; ++field) {
      check("global_", 0, field, global(field), 100);
    }

    const uint32_t action_upper[] = {30, 10, 3, 256, 4, 9, 13, 31, 10};
    auto validate_actions = [&](auto &actions, const std::string &name,
                                int rows, bool history) {
      for (int row = 0; row < rows; ++row) {
        if (actions(row, 0) > max_cards() * 2) {
          throw_domain_error(name, row, 0, actions(row, 0),
                             max_cards() * 2 + 1);
        }
        check_card_id(name, row, actions(row, 1), actions(row, 2));
        for (int field = 0; field < 9; ++field) {
          check(name, row, field + 3, actions(row, field + 3),
                action_upper[field]);
        }
        if (history) {
          check(name, row, 12, actions(row, 12), 20);
          check(name, row, 13, actions(row, 13), 12);
        }
      }
    };
    validate_actions(state["obs:actions_"_], "actions_", max_options(), false);
    validate_actions(state["obs:h_actions_"_], "h_actions_",
                     n_history_actions_, true);

    auto &mask = state["obs:mask_"_];
    for (int row = 0; row < max_cards() * 2; ++row) {
      for (int field = 0; field < 14; ++field) {
        check("mask_", row, field, mask(row, field), 2);
      }
    }

    if (!structured_enabled()) return;
    auto &visible = state["obs:visible_card_ids_"_];
    for (int row = 0; row < max_cards() * 2; ++row) {
      check_card_id("visible_card_ids_", row, visible(row, 0), visible(row, 1));
    }
    auto &tags = state["obs:effect_tags_"_];
    auto &confidence = state["obs:effect_tag_confidence_"_];
    for (int row = 0; row < max_cards() * 2; ++row) {
      for (int field = 0; field < 16; ++field) {
        check("effect_tags_", row, field, tags(row, field), 2);
        check("effect_tag_confidence_", row, field,
              confidence(row, field), 4);
      }
    }
    auto &selection = state["obs:selection_"_];
    check("selection_", 0, 0, selection(0), 30);
    for (int field : {2, 3, 9, 10, 11}) {
      check("selection_", 0, field, selection(field), 2);
    }
    check("selection_", 0, 4, selection(4), 9);
    check("selection_", 0, 13, selection(13), 4);

    auto &action_features = state["obs:action_features_"_];
    for (int row = 0; row < max_options(); ++row) {
      check("action_features_", row, 0, action_features(row, 0), 30);
      check("action_features_", row, 1, action_features(row, 1), 10);
      check("action_features_", row, 2, action_features(row, 2), 2);
      check("action_features_", row, 4, action_features(row, 4), 4);
      check("action_features_", row, 7, action_features(row, 7), 31);
      check("action_features_", row, 8, action_features(row, 8), 10);
      if (msg_ != MSG_ANNOUNCE_RACE && msg_ != MSG_ANNOUNCE_NUMBER) {
        check("action_features_", row, 11, action_features(row, 11), 4);
        check("action_features_", row, 12, action_features(row, 12), 2);
      }
      check("action_features_", row, 13, action_features(row, 13), 2);
      check("action_features_", row, 14, action_features(row, 14), 2);
    }

    const int group_references = spec_.config["max_group_references"_];
    auto &group_mask = state["obs:action_group_mask_"_];
    auto &group_refs = state["obs:action_group_refs_"_];
    for (int row = 0; row < max_options(); ++row) {
      for (int group = 0; group < 4; ++group) {
        for (int index = 0; index < group_references; ++index) {
          check("action_group_mask_", row, group * group_references + index,
                group_mask(row, group, index), 2);
          check("action_group_refs_", row, group * group_references + index,
                group_refs(row, group, index, 0), max_cards() * 2 + 1);
          check("action_group_refs_", row, group * group_references + index,
                group_refs(row, group, index, 1), 4);
        }
      }
    }
    auto &single_refs = state["obs:action_single_refs_"_];
    for (int row = 0; row < max_options(); ++row) {
      for (int index = 0; index < 4; ++index) {
        check("action_single_refs_", row, index * 3,
              single_refs(row, index, 0), 9);
        check("action_single_refs_", row, index * 3 + 1,
              single_refs(row, index, 1), max_cards() * 2 + 1);
        check("action_single_refs_", row, index * 3 + 2,
              single_refs(row, index, 2), 4);
      }
    }
    const int public_events = spec_.config["n_public_events"_];
    auto &events = state["obs:public_events_"_];
    auto &event_refs = state["obs:public_event_refs_"_];
    for (int row = 0; row < public_events; ++row) {
      check("public_events_", row, 0, events(row, 0), 13);
      // 0 is padding, 1 is the acting player, 2 is the opponent.
      check("public_events_", row, 1, events(row, 1), 3);
      for (int index = 0; index < 4; ++index) {
        check("public_event_refs_", row, index * 3,
              event_refs(row, index, 0), 9);
        check("public_event_refs_", row, index * 3 + 1,
              event_refs(row, index, 1), max_cards() * 2 + 1);
        check("public_event_refs_", row, index * 3 + 2,
              event_refs(row, index, 2), 4);
      }
    }
  }

  uint16_t visible_index(const SpecInfos &spec_infos,
                         const VisibleCardRef &ref) const {
    if (!ref.present || ref.code == 0) return 0;
    const std::string spec = ls_to_spec(
        ref.location, ref.sequence, ref.position, ref.controller != to_play_);
    auto it = spec_infos.find(spec);
    if (it == spec_infos.end()) return 0;
    const CardId expected = c_get_card_id(ref.code);
    return it->second.cid == expected ? it->second.index : 0;
  }

  void set_single_ref(TArray<uint16_t> &refs, int action, int slot,
                      uint16_t role, uint16_t index, uint16_t confidence) {
    refs(action, slot, 0) = role;
    refs(action, slot, 1) = index;
    refs(action, slot, 2) = index == 0 ? 0 : confidence;
  }

  void add_public_event(uint8_t type, PlayerId actor,
                        const VisibleCardRef &card,
                        uint8_t from = 0, uint8_t to = 0,
                        uint8_t result = 0) {
    const int capacity = spec_.config["n_public_events"_];
    if (static_cast<int>(public_events_.size()) == capacity) {
      public_events_.erase(public_events_.begin());
      ++public_event_overflow_;
    }
    public_events_.push_back(PublicEventRecord{
        type, actor, static_cast<uint8_t>(std::min(chain_depth_, 255)), from,
        to, card.position, 0, result, turn_count_, current_phase_, card});
  }

  VisibleCardRef visible_ref(const Card &card, bool expose_identity = true) const {
    return VisibleCardRef{expose_identity ? card.code_ : 0, card.controler_,
                          static_cast<uint8_t>(card.location_),
                          static_cast<uint8_t>(card.sequence_),
                          static_cast<uint8_t>(card.position_), true};
  }

  void _set_obs_public_events(State &state, const SpecInfos &spec_infos) {
    auto &events = state["obs:public_events_"_];
    auto &refs = state["obs:public_event_refs_"_];
    const int capacity = spec_.config["n_public_events"_];
    const int start = std::max(0, static_cast<int>(public_events_.size()) - capacity);
    int out = 0;
    for (int i = start; i < static_cast<int>(public_events_.size()); ++i, ++out) {
      const auto &event = public_events_[i];
      events(out, 0) = event.type;
      events(out, 1) = event.actor == to_play_ ? 1 : 2;
      events(out, 2) = event.chain_link;
      const uint8_t location_from = event.location_from & ~LOCATION_OVERLAY;
      const uint8_t location_to = event.location_to & ~LOCATION_OVERLAY;
      events(out, 3) = location_from == 0 ? 0 : location_to_id(location_from);
      events(out, 4) = location_to == 0 ? 0 : location_to_id(location_to);
      events(out, 5) = position_to_id(event.position);
      events(out, 6) = event.effect;
      events(out, 7) = event.result;
      events(out, 8) = static_cast<uint8_t>(
          std::min(std::max(0, turn_count_ - event.turn), 16));
      auto phase_it = phase2id.find(event.phase);
      events(out, 9) = phase_it == phase2id.end() ? 0 : phase_it->second;
      if (event.card.present && event.card.code != 0) {
        const CardId cid = c_get_card_id(event.card.code);
        events(out, 10) = static_cast<uint8_t>(cid >> 8);
        events(out, 11) = static_cast<uint8_t>(cid & 0xff);
      }
      const uint16_t index = visible_index(spec_infos, event.card);
      set_single_ref(refs, out, 0, 3, index, 3);
    }
  }

  void _set_obs_structured(State &state, const SpecInfos &spec_infos) {
    auto &visible_ids = state["obs:visible_card_ids_"_];
    auto &semantics = state["obs:card_semantics_"_];
    auto &tags = state["obs:effect_tags_"_];
    auto &tag_conf = state["obs:effect_tag_confidence_"_];
    if (!card_semantics_table_.empty()) {
      for (const auto &[spec, info] : spec_infos) {
        if (info.index == 0 || info.index > max_cards() * 2 || info.cid == 0) continue;
        const int i = info.index - 1;
        const uint16_t row = info.cid;
        visible_ids(i, 0) = static_cast<uint8_t>(row >> 8);
        visible_ids(i, 1) = static_cast<uint8_t>(row & 0xff);
        if ((static_cast<size_t>(row) + 1) * 32 <= card_semantics_table_.size()) {
          semantics[i].Assign(card_semantics_table_.data() + row * 32, 32);
          tags[i].Assign(effect_tags_table_.data() + row * 16, 16);
          tag_conf[i].Assign(effect_tag_confidence_table_.data() + row * 16, 16);
        }
      }
    }

    auto &selection = state["obs:selection_"_];
    selection(0) = msg_to_id(msg_);
    selection(1) = static_cast<uint8_t>(std::min(chain_depth_, 255));
    selection(2) = static_cast<uint8_t>(msg_ == MSG_SELECT_CHAIN);
    selection(3) = static_cast<uint8_t>(selection_forced_);
    selection(4) = static_cast<uint8_t>(
        msg_ == MSG_SELECT_TRIBUTE ? 7 : msg_ == MSG_SELECT_SUM ? 6 :
        (msg_ == MSG_SELECT_CARD || msg_ == MSG_SELECT_UNSELECT_CARD) ? 3 : 0);
    selection(5) = static_cast<uint8_t>(std::min(ms_min_, 255));
    selection(6) = static_cast<uint8_t>(std::min(ms_max_, 255));
    selection(7) = static_cast<uint8_t>(std::min(ms_must_, 255));
    selection(8) = static_cast<uint8_t>(std::min(static_cast<int>(ms_r_idxs_.size()), 255));
    selection(9) = static_cast<uint8_t>(selection_finishable_);
    selection(10) = static_cast<uint8_t>(
        selection_cancelable_ && !policy_back_cancel_suppressed_);
    selection(11) = static_cast<uint8_t>(ms_idx_ >= 0 && !selection_finishable_);
    const uint16_t active_index = visible_index(spec_infos, active_chain_source_);
    selection(12) = static_cast<uint8_t>(std::min<uint16_t>(active_index, 255));
    selection(13) = active_index == 0 ? 0 : 3;

    auto &features = state["obs:action_features_"_];
    auto &refs = state["obs:action_single_refs_"_];
    auto &group_refs = state["obs:action_group_refs_"_];
    auto &group_mask = state["obs:action_group_mask_"_];
    const int ngr = spec_.config["max_group_references"_];
    for (int i = 0; i < static_cast<int>(legal_actions_.size()); ++i) {
      const auto &action = legal_actions_[i];
      features(i, 0) = msg_to_id(msg_);
      features(i, 1) = static_cast<uint8_t>(action.act_);
      features(i, 2) = static_cast<uint8_t>(action.finish_);
      int structured_effect = action.effect_;
      if (structured_effect == -1) structured_effect = 0;
      else if (structured_effect == 0) structured_effect = 1;
      else if (structured_effect >= CARD_EFFECT_OFFSET)
        structured_effect = structured_effect - CARD_EFFECT_OFFSET + 2;
      else structured_effect = system_string_to_id(structured_effect);
      features(i, 3) = static_cast<uint8_t>(structured_effect);
      features(i, 4) = static_cast<uint8_t>(action.phase_);
      features(i, 5) = position_to_id(action.position_);
      features(i, 6) = static_cast<uint8_t>(
          action.number_ <= 255 ? action.number_ : 0);
      features(i, 7) = static_cast<uint8_t>(action.place_);
      features(i, 8) = attribute_to_id(action.attribute_);
      features(i, 9) = static_cast<uint8_t>(action.cid_ >> 8);
      features(i, 10) = static_cast<uint8_t>(action.cid_ & 0xff);
      features(i, 11) = action.spec_index_ == 0 ? 0 : 3;
      features(i, 12) = static_cast<uint8_t>(action.act_ == ActionAct::Cancel);
      features(i, 13) = static_cast<uint8_t>(action.finish_);
      // Use a reserved dense feature without changing the checkpoint shape.
      features(i, 14) = static_cast<uint8_t>(action.unselect_);
      if (msg_ == MSG_ANNOUNCE_RACE) {
        // Card/spec fields are unused for this message, so encode the full
        // 32-bit race mask there without changing the frozen tensor shape.
        features(i, 9) = static_cast<uint8_t>(action.race_ >> 24);
        features(i, 10) = static_cast<uint8_t>((action.race_ >> 16) & 0xff);
        features(i, 11) = static_cast<uint8_t>((action.race_ >> 8) & 0xff);
        features(i, 12) = static_cast<uint8_t>(action.race_ & 0xff);
      } else if (msg_ == MSG_ANNOUNCE_NUMBER) {
        // Preserve arbitrary protocol numbers without changing the frozen
        // tensor shape.  Card/spec fields are unused for this message.
        features(i, 9) = static_cast<uint8_t>(action.number_ >> 24);
        features(i, 10) = static_cast<uint8_t>((action.number_ >> 16) & 0xff);
        features(i, 11) = static_cast<uint8_t>((action.number_ >> 8) & 0xff);
        features(i, 12) = static_cast<uint8_t>(action.number_ & 0xff);
      }
      set_single_ref(refs, i, 0, 1, action.spec_index_, 3);
      set_single_ref(refs, i, 1, 2, active_index, 3);
      set_single_ref(refs, i, 2, 3, action.spec_index_, 3);
      if (msg_ == MSG_SELECT_CARD) {
        set_single_ref(refs, i, 3, 4, action.spec_index_, 1);
      }
      int group = msg_ == MSG_SELECT_TRIBUTE ? 2 : msg_ == MSG_SELECT_SUM ? 1 : -1;
      if (group >= 0 && action.spec_index_ != 0) {
        group_refs(i, group, 0, 0) = action.spec_index_;
        group_refs(i, group, 0, 1) = 3;
        group_mask(i, group, 0) = 1;
      }
      int selected_count = 0;
      for (int selected : ms_r_idxs_) {
        if (selected_count >= ngr) break;
        if (selected >= 0 && selected < static_cast<int>(ms_specs_.size())) {
          auto it = spec_infos.find(ms_specs_[selected]);
          if (it != spec_infos.end()) {
            group_refs(i, 3, selected_count, 0) = it->second.index;
            group_refs(i, 3, selected_count, 1) = 3;
            group_mask(i, 3, selected_count) = 1;
            ++selected_count;
          }
        }
      }
      if (static_cast<int>(ms_r_idxs_.size()) > ngr) {
        state["obs:structured_diagnostics_"_](5) = 1;
      }
    }
    state["obs:structured_diagnostics_"_](0) = static_cast<uint8_t>(last_action_overflow_);
    state["obs:structured_diagnostics_"_](6) =
        static_cast<uint8_t>(std::min<uint32_t>(public_event_overflow_, 255));
    _set_obs_public_events(state, spec_infos);
  }

  std::tuple<SpecInfos, std::vector<int>> _set_obs_cards(TArray<uint8_t> &f_cards, PlayerId to_play) {
    SpecInfos spec_infos;
    std::vector<int> loc_n_cards;
    int offset = 0;
    for (auto pi = 0; pi < 2; pi++) {
      const PlayerId player = (to_play + pi) % 2;
      const bool opponent = pi == 1;
      std::vector<std::pair<uint8_t, bool>> configs = {
          {LOCATION_DECK, true},   {LOCATION_HAND, true},
          {LOCATION_MZONE, false}, {LOCATION_SZONE, false},
          {LOCATION_GRAVE, false}, {LOCATION_REMOVED, false},
          {LOCATION_EXTRA, true},
      };
      for (auto &[location, hidden_for_opponent] : configs) {
        // check this
        if (opponent && (revealed_.size() != 0)) {
          hidden_for_opponent = false;
        }
        if (opponent && hidden_for_opponent) {
          auto n_cards = YGO_QueryFieldCount(pduel_, player, location);
          loc_n_cards.push_back(n_cards);
          for (auto i = 0; i < n_cards; i++) {
            f_cards(offset, 2) = location_to_id(location);
            f_cards(offset, 4) = 1;
            // Hidden cards still need a stable scene index because effects may
            // legally select them by location/sequence (for example oh1). The
            // identity remains zero; only the positional reference is exposed.
            const auto spec = ls_to_spec(location, i, 0, true);
            offset++;
            spec_infos[spec] = {static_cast<uint16_t>(offset), 0};
          }
        } else {
          std::vector<Card> cards = get_cards_in_location(player, location);
          int n_cards = cards.size();
          loc_n_cards.push_back(n_cards);
          for (int i = 0; i < n_cards; ++i) {
            const auto &c = cards[i];
            auto spec = c.get_spec(opponent);
            bool hide = false;
            if (opponent) {
              hide = c.position_ & POS_FACEDOWN;
              if (revealed_.find(spec) != revealed_.end()) {
                hide = false;
              }
            }
            CardId card_id = 0;
            if (!hide) {
              card_id = c_get_card_id(c.code_);
            }
            _set_obs_card_(f_cards, offset, c, hide, card_id);
            offset++;

            spec_infos[spec] = {static_cast<uint16_t>(offset), card_id};
          }
        }
      }
    }
    return {spec_infos, loc_n_cards};
  }

  void _set_obs_g_cards(TArray<uint8_t> &f_cards, PlayerId to_play) {
    int offset = 0;
    for (auto pi = 0; pi < 2; pi++) {
      const PlayerId player = (to_play + pi) % 2;
      std::vector<uint8_t> configs = {
          LOCATION_DECK, LOCATION_HAND, LOCATION_MZONE,
          LOCATION_SZONE, LOCATION_GRAVE, LOCATION_REMOVED,
          LOCATION_EXTRA,
      };
      for (auto location : configs) {
        std::vector<Card> cards = get_cards_in_location(player, location);
        int n_cards = cards.size();
        for (int i = 0; i < n_cards; ++i) {
          const auto &c = cards[i];
          CardId card_id = c_get_card_id(c.code_);
          _set_obs_card_(f_cards, offset, c, false, card_id, false);
          offset++;
          if (offset == (spec_.config["max_cards"_] * 2 - 1)) {
            return;
          }
        }
      }
    }
  }

  std::tuple<SpecInfos, std::vector<int>> _set_obs_mask(TArray<uint8_t> &mask, PlayerId to_play) {
    SpecInfos spec_infos;
    std::vector<int> loc_n_cards;
    int offset = 0;
    for (auto pi = 0; pi < 2; pi++) {
      const PlayerId player = (to_play + pi) % 2;
      const bool opponent = pi == 1;
      std::vector<std::pair<uint8_t, bool>> configs = {
          {LOCATION_DECK, true},   {LOCATION_HAND, true},
          {LOCATION_MZONE, false}, {LOCATION_SZONE, false},
          {LOCATION_GRAVE, false}, {LOCATION_REMOVED, false},
          {LOCATION_EXTRA, true},
      };
      for (auto &[location, hidden_for_opponent] : configs) {
        // check this
        if (opponent && (revealed_.size() != 0)) {
          hidden_for_opponent = false;
        }
        if (opponent && hidden_for_opponent) {
          auto n_cards = YGO_QueryFieldCount(pduel_, player, location);
          loc_n_cards.push_back(n_cards);
          for (auto i = 0; i < n_cards; i++) {
            mask(offset, 1) = 1;
            mask(offset, 3) = 1;
            const auto spec = ls_to_spec(location, i, 0, true);
            offset++;
            spec_infos[spec] = {static_cast<uint16_t>(offset), 0};
          }
        } else {
          std::vector<Card> cards = get_cards_in_location(player, location);
          int n_cards = cards.size();
          loc_n_cards.push_back(n_cards);
          for (int i = 0; i < n_cards; ++i) {
            const auto &c = cards[i];
            auto spec = c.get_spec(opponent);
            bool hide = false;
            if (opponent) {
              hide = c.position_ & POS_FACEDOWN;
              if (revealed_.find(spec) != revealed_.end()) {
                hide = false;
              }
            }
            CardId card_id = 0;
            if (!hide) {
              card_id = c_get_card_id(c.code_);
            }
            _set_obs_mask_(mask, offset, c, hide);
            offset++;

            spec_infos[spec] = {static_cast<uint16_t>(offset), card_id};
          }
        }
      }
    }
    return {spec_infos, loc_n_cards};
  }

  void _set_obs_card_(TArray<uint8_t> &f_cards, int offset, const Card &c,
                      bool hide, CardId card_id = 0, bool global = false) {
    // check offset exceeds max_cards
    uint8_t location = c.location_;
    bool overlay = location & LOCATION_OVERLAY;
    if (overlay) {
      location = location & 0x7f;
    }
    if (overlay) {
      hide = false;
    }

    if (!hide) {
      f_cards(offset, 0) = static_cast<uint8_t>(card_id >> 8);
      f_cards(offset, 1) = static_cast<uint8_t>(card_id & 0xff);
    }
    f_cards(offset, 2) = location_to_id(location);

    uint8_t seq = 0;
    if (location == LOCATION_MZONE || location == LOCATION_SZONE ||
        location == LOCATION_GRAVE) {
      seq = c.sequence_ + 1;
    }
    f_cards(offset, 3) = seq;
    f_cards(offset, 4) = global ? c.controler_ : ((c.controler_ != to_play_) ? 1 : 0);
    if (overlay) {
      f_cards(offset, 5) = position_to_id(POS_FACEUP);
      f_cards(offset, 6) = 1;
    } else {
      if (location == LOCATION_DECK || location == LOCATION_HAND || location == LOCATION_EXTRA) {
        if (hide || (c.position_ & POS_FACEDOWN)) {
          f_cards(offset, 5) = position_to_id(POS_FACEDOWN);
        }
        // else {
        //   fmt::println("location: {}, position: {}", location2str.at(location), position_to_string(c.position_));
        // }
      } else {
        f_cards(offset, 5) = position_to_id(c.position_);
      }
    }
    if (!hide) {
      f_cards(offset, 7) = attribute_to_id(c.attribute_);
      f_cards(offset, 8) = race_to_id(c.race_);
      // Effects can raise a card above the embedding's represented range.
      // Index 13 is the explicit 13-or-higher overflow bucket.
      f_cards(offset, 9) = static_cast<uint8_t>(std::min(c.level_, uint32_t(13)));
      f_cards(offset, 10) = std::min(c.counter_, static_cast<uint32_t>(15));
      f_cards(offset, 11) = static_cast<uint8_t>((c.status_ & (STATUS_DISABLED | STATUS_FORBIDDEN)) != 0);
      auto [atk1, atk2] = float_transform(c.attack_);
      f_cards(offset, 12) = atk1;
      f_cards(offset, 13) = atk2;

      auto [def1, def2] = float_transform(c.defense_);
      f_cards(offset, 14) = def1;
      f_cards(offset, 15) = def2;

      auto type_ids = type_to_ids(c.type_);
      for (int j = 0; j < type_ids.size(); ++j) {
        f_cards(offset, 16 + j) = type_ids[j];
      }
    }
  }

  void _set_obs_mask_(TArray<uint8_t> &mask, int offset, const Card &c,
                      bool hide, CardId card_id = 0, bool global = false) {
    // check offset exceeds max_cards
    uint8_t location = c.location_;
    bool overlay = location & LOCATION_OVERLAY;
    if (overlay) {
      location = location & 0x7f;
    }
    if (overlay) {
      hide = false;
    }

    if (!hide) {
      if (card_id != 0) {
        mask(offset, 0) = 1;
      }
    }
    mask(offset, 1) = 1;

    if (location == LOCATION_MZONE || location == LOCATION_SZONE ||
        location == LOCATION_GRAVE) {
      mask(offset, 2) = 1;
    }
    mask(offset, 3) = 1;
    if (overlay) {
      mask(offset, 4) = 1;
      mask(offset, 5) = 1;
    } else {
      if (location == LOCATION_DECK || location == LOCATION_HAND || location == LOCATION_EXTRA) {
        if (hide || (c.position_ & POS_FACEDOWN)) {
          mask(offset, 4) = 1;
        }
      } else {
        mask(offset, 4) = 1;
      }
    }
    if (!hide) {
      mask(offset, 6) = 1;
      mask(offset, 7) = 1;
      mask(offset, 8) = 1;
      mask(offset, 9) = 1;
      mask(offset, 10) = 1;
      mask(offset, 11) = 1;
      mask(offset, 12) = 1;
      mask(offset, 13) = 1;
    }
  }

  void _set_obs_global(TArray<uint8_t> &feat, PlayerId player, const std::vector<int> &loc_n_cards) {
    uint8_t me = player;
    uint8_t op = 1 - player;

    auto [me_lp_1, me_lp_2] = float_transform(lp_[me]);
    feat(0) = me_lp_1;
    feat(1) = me_lp_2;

    auto [op_lp_1, op_lp_2] = float_transform(lp_[op]);
    feat(2) = op_lp_1;
    feat(3) = op_lp_2;

    feat(4) = std::min(turn_count_, 16);
    feat(5) = phase_to_id(current_phase_);
    feat(6) = (me == 0) ? 1 : 0;
    feat(7) = (me == tp_) ? 1 : 0;

    for (int i = 0; i < loc_n_cards.size(); i++) {
      feat(8 + i) = static_cast<uint8_t>(loc_n_cards[i]);
    }
  }

  const SpecInfo& find_spec_info(SpecInfos &spec_infos, const std::string &spec) {
    auto it = spec_infos.find(spec);
    if (it == spec_infos.end()) {
      // This lookup feeds model-visible card references. A fabricated fallback
      // silently corrupts observations, so fail at the protocol boundary with
      // the complete diagnostic state instead.
      show_deck(0);
      show_deck(1);
      show_buffer();
      show_turn();
      fmt::println("MS: idx: {}, mode: {}, min: {}, max: {}, must: {}, specs: {}, combs: {}", ms_idx_, ms_mode_, ms_min_, ms_max_, ms_must_, ms_specs_, ms_combs_);
      fmt::println("Spec: {}, Spec2index:", spec);
      for (auto &[k, v] : spec_infos) {
        fmt::print("{}: {} {}, ", k, v.index, v.cid);
      }
      fmt::print("\n");
      throw std::runtime_error(fmt::format(
          "[protocol boundary] observation spec not found: {}", spec));
    }
    return it->second;
  }

  void _set_obs_action_spec(
    TArray<uint8_t> &feat, int i, int idx) {
    feat(i, 0) = static_cast<uint8_t>(idx);
  }

  void _set_obs_action_card_id(
    TArray<uint8_t> &feat, int i, CardId cid) {
    feat(i, 1) = static_cast<uint8_t>(cid >> 8);
    feat(i, 2) = static_cast<uint8_t>(cid & 0xff);
  }

  void _set_obs_action_msg(TArray<uint8_t> &feat, int i, int msg) {
    const uint8_t encoded = msg_to_id(msg);
    if (encoded >= 30) {
      throw std::runtime_error(
          fmt::format("[action encoding] message id out of range: {}", encoded));
    }
    feat(i, 3) = encoded;
  }

  void _set_obs_action_act(TArray<uint8_t> &feat, int i, ActionAct act) {
    const uint8_t encoded = static_cast<uint8_t>(act);
    if (encoded >= 10) {
      throw std::runtime_error(
          fmt::format("[action encoding] action id out of range: {}", encoded));
    }
    feat(i, 4) = encoded;
  }

  void _set_obs_action_finish(TArray<uint8_t> &feat, int i) {
    feat(i, 5) = 1;
  }

  void _set_obs_action_effect(TArray<uint8_t> &feat, int i, int effect) {
    // 0: None
    // 1: default
    // 2-15: card effect
    // 16+: system
    if (effect == -1) {
      effect = 0;
    } else if (effect == 0) {
      effect = 1;
    } else if (effect >= CARD_EFFECT_OFFSET) {
      effect = effect - CARD_EFFECT_OFFSET + 2;
    } else {
      effect = system_string_to_id(effect);
    }
    feat(i, 6) = static_cast<uint8_t>(effect);
  }

  void _set_obs_action_phase(TArray<uint8_t> &feat, int i, ActionPhase phase){
    feat(i, 7) = static_cast<uint8_t>(phase);
  }

  void _set_obs_action_position(TArray<uint8_t> &feat, int i, uint8_t position) {
    const uint8_t encoded = position_to_id(position);
    // The legacy policy has nine frozen position-embedding rows.  Structured
    // observations retain the extended composite-mask id in action_features_,
    // while legacy actions use the generic row instead of indexing out of
    // bounds and poisoning the complete forward pass with NaNs.
    feat(i, 8) = encoded < 9 ? encoded : 0;
  }

  void _set_obs_action_number(TArray<uint8_t> &feat, int i, uint32_t number) {
    // The legacy model has thirteen frozen number rows (0..12).  Larger
    // values remain fully represented in Structured-lite action_features_.
    feat(i, 9) = static_cast<uint8_t>(number <= 12 ? number : 0);
  }

  void _set_obs_action_place(TArray<uint8_t> &feat, int i, ActionPlace place) {
    feat(i, 10) = static_cast<uint8_t>(place);
  }

  void _set_obs_action_attrib(TArray<uint8_t> &feat, int i, uint8_t attrib) {
    feat(i, 11) = attribute_to_id(attrib);
  }

  void _set_obs_action_race(TArray<uint8_t> &feat, int i, uint32_t race) {
    // The legacy tensor has no race-mask field.  In particular, bytes 3 and 4
    // are categorical message/action ids and must never be overwritten with
    // arbitrary mask bytes.  Structured-lite stores the complete 32-bit mask
    // in action_features_[9:13], so leave the legacy action unchanged.
    (void)feat;
    (void)i;
    (void)race;
  }

  void _set_obs_action(TArray<uint8_t> &feat, int i, const LegalAction &action) {
    auto msg = action.msg_;
    _set_obs_action_msg(feat, i, msg);
    _set_obs_action_card_id(feat, i, action.cid_);
    if (msg == MSG_SELECT_CARD || msg == MSG_SELECT_TRIBUTE ||
        msg == MSG_SELECT_SUM || msg == MSG_SELECT_UNSELECT_CARD ||
        msg == MSG_SORT_CARD) {
      if (action.act_ == ActionAct::Cancel) {
        _set_obs_action_act(feat, i, action.act_);
      } else if (action.finish_) {
        _set_obs_action_finish(feat, i);
      } else {
        _set_obs_action_spec(feat, i, action.spec_index_);
      }
    } else if (msg == MSG_SELECT_POSITION) {
      _set_obs_action_position(feat, i, action.position_);
    } else if (msg == MSG_SELECT_EFFECTYN) {
      _set_obs_action_spec(feat, i, action.spec_index_);
      _set_obs_action_act(feat, i, action.act_);
      _set_obs_action_effect(feat, i, action.effect_);
    } else if (msg == MSG_SELECT_YESNO || msg == MSG_SELECT_OPTION) {
      _set_obs_action_act(feat, i, action.act_);
      _set_obs_action_effect(feat, i, action.effect_);
    } else if (
      msg == MSG_SELECT_BATTLECMD ||
      msg == MSG_SELECT_IDLECMD ||
      msg == MSG_SELECT_CHAIN) {
      _set_obs_action_phase(feat, i, action.phase_);
      _set_obs_action_spec(feat, i, action.spec_index_);
      _set_obs_action_act(feat, i, action.act_);
      _set_obs_action_effect(feat, i, action.effect_);
    } else if (msg == MSG_SELECT_PLACE || msg == MSG_SELECT_DISFIELD) {
      _set_obs_action_place(feat, i, action.place_);
    } else if (msg == MSG_ANNOUNCE_CARD) {
      // card id, already set
    } else if (msg == MSG_ANNOUNCE_ATTRIB) {
      _set_obs_action_attrib(feat, i, action.attribute_);
    } else if (msg == MSG_ANNOUNCE_RACE) {
      _set_obs_action_race(feat, i, action.race_);
    } else if (msg == MSG_ANNOUNCE_NUMBER ||
               msg == MSG_SELECT_COUNTER ||
               msg == MSG_ROCK_PAPER_SCISSORS) {
      _set_obs_action_number(feat, i, action.number_);
    } else {
      throw std::runtime_error("Unsupported message " + msg_to_string(msg));
    }
  }

  CardId spec_to_card_id(const std::string &spec, PlayerId player) {
    int offset = 0;
    bool opponent = false;
    if (spec[0] == 'o') {
      player = 1 - player;
      opponent = true;
      offset++;
    }
    auto [loc, seq, pos] = spec_to_ls(spec.substr(offset));
    if (opponent) {
      bool hidden_for_opponent = true;
      if (
        loc == LOCATION_MZONE || loc == LOCATION_SZONE ||
        loc == LOCATION_GRAVE || loc == LOCATION_REMOVED) {
        hidden_for_opponent = false;
      }
      if (revealed_.size() != 0) {
        hidden_for_opponent = false;
      }
      if (hidden_for_opponent) {
        return 0;
      }
      Card c = get_card(player, loc, seq);
      bool hide = c.position_ & POS_FACEDOWN;
      if (revealed_.find(spec) != revealed_.end()) {
        hide = false;
      }
      CardId card_id = 0;
      if (!hide) {
        card_id = c_get_card_id(c.code_);
      }
    }
    return c_get_card_id(get_card_code(player, loc, seq));
  }

  void _set_obs_actions(TArray<uint8_t> &feat, const std::vector<LegalAction> &actions) {
    for (int i = 0; i < actions.size(); ++i) {
      _set_obs_action(feat, i, actions[i]);
    }
  }


  void str_to_uint16(const char* src, uint16_t* dest) {
      for (int i = 0; i < strlen(src); i += 1) {
        dest[i] = src[i];
      }

      // Add null terminator
      dest[strlen(src) + 1] = '\0';
  }

  void ReplayWriteInt8(int8_t value) {
    fwrite(&value, sizeof(value), 1, fp_);
  }

  void ReplayWriteInt32(int32_t value) {
    fwrite(&value, sizeof(value), 1, fp_);
  }

  // ygopro-core API
  intptr_t YGO_CreateDuel(uint32_t seed) {
    std::mt19937 rnd(seed);
    // return create_duel(rnd());
    duel* pduel = new duel();
    pduel->random.reset(rnd());
    return (intptr_t)pduel;
  }

  void YGO_SetPlayerInfo(intptr_t pduel, int32 playerid, int32 lp, int32 startcount, int32 drawcount) const {
    set_player_info(pduel, playerid, lp, startcount, drawcount);
  }

  void YGO_NewCard(intptr_t pduel, uint32 code, uint8 owner, uint8 playerid, uint8 location, uint8 sequence, uint8 position) const {
    new_card(pduel, code, owner, playerid, location, sequence, position);
  }

  void YGO_StartDuel(intptr_t pduel, int32 options) const {
    start_duel(pduel, options);
  }

  void YGO_EndDuel(intptr_t pduel) const {
    // end_duel(pduel);
    duel* pd = (duel*)pduel;
    delete pd;
  }

  int32 YGO_GetMessage(intptr_t pduel, byte* buf) {
    return get_message(pduel, buf);
  }

  uint32 YGO_Process(intptr_t pduel) {
    return process(pduel);
  }

  int32 YGO_QueryCard(intptr_t pduel, uint8 playerid, uint8 location, uint8 sequence, int32 query_flag, byte* buf) {
    return query_card(pduel, playerid, location, sequence, query_flag, buf, 0);
  }

  int32 YGO_QueryFieldCount(intptr_t pduel, uint8 playerid, uint8 location) {
    return query_field_count(pduel, playerid, location);
  }

  int32 YGO_QueryFieldCard(intptr_t pduel, uint8 playerid, uint8 location, uint32 query_flag, byte* buf) {
    return query_field_card(pduel, playerid, location, query_flag, buf, 0);
  }

  void YGO_SetResponsei(intptr_t pduel, int32 value) {
    if (record_) {
      ReplayWriteInt8(4);
      ReplayWriteInt32(value);
    }
    set_responsei(pduel, value);
  }

  void YGO_SetResponseb(intptr_t pduel, byte* buf) {
    if (record_) {
      switch (msg_) {
        case MSG_SORT_CARD:
          ReplayWriteInt8(buf[0] == 0xff ? 1 : n_sort_cards_);
          fwrite(buf, buf[0] == 0xff ? 1 : n_sort_cards_, 1, fp_);
          break;
        case MSG_SELECT_COUNTER:
          ReplayWriteInt8(2 * n_counters_);
          fwrite(buf, 2 * n_counters_, 1, fp_);
          break;
        case MSG_SELECT_PLACE:
        case MSG_SELECT_DISFIELD:
          ReplayWriteInt8(3 * n_places_);
          fwrite(buf, 3 * n_places_, 1, fp_);
          break;
        default:
          ReplayWriteInt8(buf[0] + 1);
          fwrite(buf, buf[0] + 1, 1, fp_);
          break;
      }
    }
    set_responseb(pduel, buf);
  }

  // ygopro-core API

  void show_decision(int idx) {
    std::string s;
    const auto& a = legal_actions_[idx];
    if (!a.spec_.empty()) {
      s = a.spec_;
    } else if (a.place_ != ActionPlace::None) {
      s = action_place_to_string(a.place_);
    } else if (a.position_ != 0) {
      s = position_to_string(a.position_);
    } else {
      s = fmt::format("{}", a);
    }
    fmt::print("Player {} chose \"{}\" in {}\n", to_play_, s, legal_actions_);
  }

  std::tuple<std::vector<CardCode>, std::vector<CardCode>, std::string>
  load_deck(
    intptr_t pduel, PlayerId player, std::mt19937& gen, bool shuffle = true) {
    std::string deck_name = player == 0 ? deck1_ : deck2_;

    if (deck_name == "manifest") {
      if (sampler_clusters_.empty()) {
        throw std::runtime_error("manifest deck requested without deck_sampling_manifest");
      }
      deck_name = sample_manifest_deck(player);
    } else if (deck_name == "random") {
      // generate random deck name
      std::uniform_int_distribution<uint64_t> dist_int(0,
                                                       deck_names_.size() - 1);
      deck_name = deck_names_[dist_int(gen)];
    }

    std::vector<CardCode> main_deck = main_decks_.at(deck_name);
    std::vector<CardCode> extra_deck = extra_decks_.at(deck_name);

    if (verbose_) {
      fmt::println("{} {}: {}, main({}), extra({})", player, nickname_[player],
        deck_name, main_deck.size(), extra_deck.size());
    }

    if (shuffle) {
      std::shuffle(main_deck.begin(), main_deck.end(), gen);
    }

    // add main deck in reverse order following ygopro
    // but since we have shuffled deck, so just add in order

    for (int i = 0; i < main_deck.size(); i++) {
      YGO_NewCard(pduel, main_deck[i], player, player, LOCATION_DECK, 0,
               POS_FACEDOWN_DEFENSE);
    }

    // add extra deck in reverse order following ygopro
    for (int i = int(extra_deck.size()) - 1; i >= 0; --i) {
      YGO_NewCard(pduel, extra_deck[i], player, player, LOCATION_EXTRA, 0,
               POS_FACEDOWN_DEFENSE);
    }

    return {main_deck, extra_deck, deck_name};
  }

  void next() {
    while (duel_started_) {
      if (eng_flag_ == PROCESSOR_END) {
        break;
      }
      uint32_t res = YGO_Process(pduel_);
      dl_ = res & PROCESSOR_BUFFER_LEN;
      eng_flag_ = res & PROCESSOR_FLAG;

      if (dl_ == 0) {
        continue;
      }
      YGO_GetMessage(pduel_, data_);
      dp_ = 0;
      while ((dp_ != dl_) || (ms_idx_ != -1)) {
        int message_start = dp_;
        if (ms_idx_ != -1) {
          handle_multi_select();
        } else {
          handle_message();
        }
        filter_policy_navigational_back();
        if (legal_actions_.empty()) {
          if (ms_idx_ != -1) {
            throw std::runtime_error(
                "staged protocol selection has no legal actions");
          }
          // Some prompts are resolved locally by the parser or a staged
          // selection (for example a forced counter allocation).
          if (play_mode_ == kWindBot && message_start < dp_ &&
              !windbot_response_message(msg_)) {
            windbot_game_message(data_ + message_start, dp_ - message_start);
          }
          continue;
        }
        if (play_mode_ == kWindBot && to_play_ != ai_player_) {
          if (ms_idx_ == -1 || message_start < dp_) {
            windbot_game_message(data_ + message_start, dp_ - message_start);
          }
          windbot_apply_response();
          continue;
        }
        if ((play_mode_ == kSelfPlay) || (to_play_ == ai_player_)) {
          if (legal_actions_.size() == 1) {
            if (!callback_) {
              throw std::runtime_error(fmt::format(
                  "Missing action callback for message {} ({})",
                  msg_, msg_to_string(msg_)));
            }
            callback_(0);
            auto la = legal_actions_[0];
            la.msg_ = msg_;
            if (la.cid_ == 0 && !la.spec_.empty()) {
              la.cid_ = spec_to_card_id(la.spec_, to_play_);
            }
            update_history_actions(to_play_, la);
            if (verbose_) {
              show_decision(0);
            }
          } else {
            return;
          }
        } else {
          auto idx = players_[to_play_]->think(legal_actions_);
          callback_(idx);
          if (verbose_) {
            show_decision(idx);
          }
        }
      }
    }
    done_ = true;
    legal_actions_.clear();
  }

  void require_message_bytes(int count) const {
    if (count < 0 || dp_ < 0 || dl_ < dp_ || count > dl_ - dp_) {
      throw std::runtime_error(fmt::format(
          "Truncated message {}: need {} byte(s), remaining {}, dp {}, length {}",
          msg_to_string(msg_), count, std::max(0, dl_ - dp_), dp_, dl_));
    }
  }

  void skip_message_bytes(int count) {
    require_message_bytes(count);
    dp_ += count;
  }

  uint8_t read_u8() {
    require_message_bytes(1);
    return data_[dp_++];
  }

  uint16_t read_u16() {
    require_message_bytes(2);
    uint16_t v = 0;
    std::memcpy(&v, data_ + dp_, sizeof(v));
    dp_ += 2;
    return v;
  }

  uint32 read_u32() {
    require_message_bytes(4);
    uint32 v = 0;
    std::memcpy(&v, data_ + dp_, sizeof(v));
    dp_ += 4;
    return v;
  }

  uint32 q_read_u8() {
    uint8_t v = *reinterpret_cast<uint8_t *>(query_buf_ + qdp_);
    qdp_ += 1;
    return v;
  }

  uint32 q_read_u32() {
    uint32_t v = *reinterpret_cast<uint32_t *>(query_buf_ + qdp_);
    qdp_ += 4;
    return v;
  }

  CardCode get_card_code(PlayerId player, uint8_t loc, uint8_t seq) {
    int32_t flags = QUERY_CODE;
    int32_t bl = YGO_QueryCard(pduel_, player, loc, seq, flags, query_buf_);
    qdp_ = 0;
    if (bl <= 0) {
      throw std::runtime_error("[get_card_code] Invalid card");
    }
    qdp_ += 8;
    return q_read_u32();
  }

  Card get_card(PlayerId player, uint8_t loc, uint8_t seq) {
    int32_t flags = QUERY_CODE | QUERY_ATTACK | QUERY_DEFENSE | QUERY_POSITION |
                    QUERY_LEVEL | QUERY_RANK | QUERY_LSCALE | QUERY_RSCALE |
                    QUERY_LINK;
    int32_t bl = YGO_QueryCard(pduel_, player, loc, seq, flags, query_buf_);
    qdp_ = 0;
    if (bl <= 0) {
      show_deck(0);
      show_deck(1);
      show_turn();
      show_buffer();
      auto s = fmt::format("[get_card] Invalid card (bl <= 0), player: {}, loc: {}, seq: {}", player, loc, seq);
      throw std::runtime_error(s);
    }
    uint32_t f = q_read_u32();
    if (f == LEN_EMPTY) {
      return Card();
    }
    f = q_read_u32();
    CardCode code = q_read_u32();
    Card c = c_get_card(code);
    uint32_t position = q_read_u32();
    c.set_location(position);
    uint32_t level = q_read_u32();
    if ((level & 0xff) > 0) {
      c.level_ = level & 0xff;
    }
    uint32_t rank = q_read_u32();
    if ((rank & 0xff) > 0) {
      c.level_ = rank & 0xff;
    }
    c.attack_ = q_read_u32();
    c.defense_ = q_read_u32();
    c.lscale_ = q_read_u32();
    c.rscale_ = q_read_u32();
    uint32_t link = q_read_u32();
    uint32_t link_marker = q_read_u32();
    if ((link & 0xff) > 0) {
      c.level_ = link & 0xff;
    }
    if (link_marker > 0) {
      c.defense_ = link_marker;
    }
    return c;
  }

  std::vector<Card> get_cards_in_location(PlayerId player, uint8_t loc) {
    int32_t flags = QUERY_CODE | QUERY_POSITION | QUERY_LEVEL | QUERY_RANK |
                    QUERY_ATTACK | QUERY_DEFENSE | QUERY_EQUIP_CARD |
                    QUERY_OVERLAY_CARD | QUERY_COUNTERS | QUERY_STATUS |
                    QUERY_LSCALE | QUERY_RSCALE | QUERY_LINK;
    int32_t bl = YGO_QueryFieldCard(pduel_, player, loc, flags, query_buf_);
    qdp_ = 0;
    std::vector<Card> cards;
    while (true) {
      if (qdp_ >= bl) {
        break;
      }
      uint32_t f = q_read_u32();
      if (f == LEN_EMPTY) {
        continue;
        ;
      }
      f = q_read_u32();
      CardCode code = q_read_u32();
      Card c = c_get_card(code);

      uint8_t controller = q_read_u8();
      uint8_t location = q_read_u8();
      uint8_t sequence = q_read_u8();
      uint8_t position = q_read_u8();
      c.controler_ = controller;
      c.location_ = location;
      c.sequence_ = sequence;
      c.position_ = position;

      uint32_t level = q_read_u32();
      if ((level & 0xff) > 0) {
        c.level_ = level & 0xff;
      }
      uint32_t rank = q_read_u32();
      if ((rank & 0xff) > 0) {
        c.level_ = rank & 0xff;
      }
      c.attack_ = q_read_u32();
      c.defense_ = q_read_u32();

      // TODO(2): equip_target
      if (f & QUERY_EQUIP_CARD) {
        q_read_u32();
      }

      uint32_t n_xyz = q_read_u32();
      for (int i = 0; i < n_xyz; ++i) {
        auto code = q_read_u32();
        Card c_ = c_get_card(code);
        c_.controler_ = controller;
        c_.location_ = location | LOCATION_OVERLAY;
        c_.sequence_ = sequence;
        c_.position_ = i;
        cards.push_back(c_);
      }

      // TODO(2): counters
      uint32_t n_counters = q_read_u32();
      for (int i = 0; i < n_counters; ++i) {
        if (i == 0) {
          c.counter_ = q_read_u32();
        }
        else {
          q_read_u32();
        }
      }

      c.status_ = q_read_u32();
      c.lscale_ = q_read_u32();
      c.rscale_ = q_read_u32();

      uint32_t link = q_read_u32();
      uint32_t link_marker = q_read_u32();
      if ((link & 0xff) > 0) {
        c.level_ = link & 0xff;
      }
      if (link_marker > 0) {
        c.defense_ = link_marker;
      }
      cards.push_back(c);
    }
    return cards;
  }

  std::vector<Card> read_cardlist(bool extra = false, bool extra8 = false) {
    std::vector<Card> cards;
    auto count = read_u8();
    cards.reserve(count);
    for (int i = 0; i < count; ++i) {
      auto code = read_u32();
      auto controller = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      auto card = get_card(controller, loc, seq);
      if (extra) {
        if (extra8) {
          card.data_ = read_u8();
        } else {
          card.data_ = read_u32();
        }
      }
      cards.push_back(card);
    }
    return cards;
  }

  std::vector<IdleCardSpec> read_cardlist_spec(PlayerId player, bool extra = false, bool extra8 = false) {
    std::vector<IdleCardSpec> card_specs;
    auto count = read_u8();
    card_specs.reserve(count);
    for (int i = 0; i < count; ++i) {
      CardCode code = read_u32();
      auto controller = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      uint32_t data = 0;
      if (extra) {
        if (extra8) {
          data = read_u8();
        } else {
          data = read_u32();
        }
      }
      card_specs.push_back({code, ls_to_spec(loc, seq, 0, player != controller), data});
    }
    return card_specs;
  }

  std::tuple<CardCode, int> unpack_desc(CardCode code, uint32_t desc) {
    if (desc < DESCRIPTION_LIMIT) {
      return {0, desc};
    }
    CardCode code_ = desc >> 4;
    int idx = desc & 0xf;
    if (idx < 0 || idx >= 14) {
      fmt::print("Code: {}, Code_: {}, Desc: {}\n", code, code_, desc);
      show_deck(0);
      show_deck(1);
      show_buffer();
      show_turn();
      throw std::runtime_error("Invalid effect index: " + std::to_string(idx));
    }
    return {code_, idx + CARD_EFFECT_OFFSET};
  }

  std::string cardlist_info_for_player(const Card &card, PlayerId pl) {
    std::string spec = card.get_spec(pl);
    if (card.location_ == LOCATION_DECK) {
      spec = "deck";
    }
    if ((card.controler_ != pl) && (card.position_ & POS_FACEDOWN)) {
      return position_to_string(card.position_) + "card (" + spec + ")";
    }
    return card.name_ + " (" + spec + ")";
  }

  // This function does the following:
  // 1. read msg_ from data_ and update dp_
  // 2. (optional) print information if verbose_ is true
  // 3. update to_play_ and options_ if need action
  void handle_message() {
    msg_ = int(data_[dp_++]);
    legal_actions_ = {};
    selection_forced_ = false;
    selection_finishable_ = false;
    selection_cancelable_ = false;
    policy_back_cancel_suppressed_ = false;

    if (verbose_) {
      fmt::println("Message {}, length {}, dp {}", msg_to_string(msg_), dl_, dp_);
    }

    if (msg_ == MSG_RELOAD_FIELD) {
      const auto duel_rule = read_u8();
      for (int player = 0; player < 2; ++player) {
        lp_[player] = static_cast<int>(read_u32());
        for (int zone = 0; zone < 7; ++zone) {
          const auto occupied = read_u8();
          if (occupied > 1) {
            throw std::runtime_error("Invalid MSG_RELOAD_FIELD monster-zone flag");
          }
          if (occupied) {
            read_u8();
            read_u8();
          }
        }
        for (int zone = 0; zone < 8; ++zone) {
          const auto occupied = read_u8();
          if (occupied > 1) {
            throw std::runtime_error("Invalid MSG_RELOAD_FIELD spell-zone flag");
          }
          if (occupied) read_u8();
        }
        skip_message_bytes(6);
      }
      const auto chain_count = read_u8();
      for (int chain = 0; chain < chain_count; ++chain) {
        read_u32();
        read_u32();
        read_u8();
        read_u8();
        read_u8();
        read_u32();
      }
      chain_depth_ = chain_count;
      active_chain_source_ = VisibleCardRef{};
      revealed_.clear();
      if (verbose_) {
        for (auto &pl : players_) {
          pl->notify(fmt::format(
              "Reloaded field snapshot for duel rule {} with {} chain link(s).",
              duel_rule, chain_count));
        }
      }
    } else if (msg_ == MSG_AI_NAME || msg_ == MSG_SHOW_HINT) {
      const auto length = read_u16();
      require_message_bytes(static_cast<int>(length) + 1);
      std::string value(reinterpret_cast<const char *>(data_ + dp_), length);
      skip_message_bytes(length);
      const auto terminator = read_u8();
      if (terminator != 0) {
        throw std::runtime_error(fmt::format(
            "Invalid {} string terminator", msg_to_string(msg_)));
      }
      if (verbose_) {
        if (msg_ == MSG_AI_NAME) players_[1]->notify("AI name: " + value);
        else for (auto &pl : players_) pl->notify(value);
      }
    } else if (msg_ == MSG_DRAW) {
      auto player = read_u8();
      auto drawed = read_u8();
      std::vector<uint32> codes;
      for (int i = 0; i < drawed; ++i) {
        uint32 code = read_u32();
        codes.push_back(code & 0x7fffffff);
      }
      add_public_event(kEventDraw, player, VisibleCardRef{}, 0, 0, drawed);
      if (!verbose_) return;
      const auto &pl = players_[player];
      pl->notify(fmt::format("Drew {} cards:", drawed));
      for (int i = 0; i < drawed; ++i) {
        const auto &c = c_get_card(codes[i]);
        pl->notify(fmt::format("{}: {}", i + 1, c.name_));
      }
      const auto &op = players_[1 - player];
      op->notify(fmt::format("Opponent drew {} cards.", drawed));
    } else if (msg_ == MSG_NEW_TURN) {
      tp_ = int(read_u8());
      turn_count_++;
      if (!verbose_) {
        return;
      }
      auto& player = players_[tp_];
      player->notify("Your turn.");
      players_[1 - tp_]->notify(fmt::format("{}'s turn.", player->nickname_));
    } else if (msg_ == MSG_NEW_PHASE) {
      current_phase_ = int(read_u16());
      if (!verbose_) {
        return;
      }
      auto phase_str = phase_to_string(current_phase_);
      for (int i = 0; i < 2; ++i) {
        players_[i]->notify(fmt::format("Entering {} phase.", phase_str));
      }
    } else if (msg_ == MSG_MOVE) {
      CardCode code = read_u32();
      uint32_t location = read_u32();
      uint32_t newloc = read_u32();
      uint32_t reason = read_u32();
      Card card = c_get_card(code);
      card.set_location(location);
      Card cnew = c_get_card(code);
      cnew.set_location(newloc);
      const bool public_identity =
          !(card.position_ & POS_FACEDOWN) || !(cnew.position_ & POS_FACEDOWN) ||
          cnew.location_ == LOCATION_GRAVE || cnew.location_ == LOCATION_REMOVED;
      add_public_event(kEventMovement, card.controler_, visible_ref(cnew, public_identity),
                       card.location_, cnew.location_);
      if (card.location_ == LOCATION_DECK && cnew.location_ == LOCATION_HAND) {
        add_public_event(kEventSearch, card.controler_, VisibleCardRef{},
                         card.location_, cnew.location_, 1);
      }
      if ((reason & REASON_DESTROY) && (card.location_ != cnew.location_)) {
        add_public_event(kEventDestruction, card.controler_,
                         visible_ref(cnew, public_identity), card.location_,
                         cnew.location_, 1);
      }
      if (!verbose_) return;
      auto& pl = players_[card.controler_];
      auto& op = players_[1 - card.controler_];

      auto plspec = card.get_spec(false);
      auto opspec = card.get_spec(true);
      auto plnewspec = cnew.get_spec(false);
      auto opnewspec = cnew.get_spec(true);

      auto getspec = [&](auto& p) { return p.get() == pl.get() ? plspec : opspec; };
      auto getnewspec = [&](auto& p) {
        return p.get() == pl.get() ? plnewspec : opnewspec;
      };
      bool card_visible = true;
      if ((card.position_ & POS_FACEDOWN) && (cnew.position_ & POS_FACEDOWN)) {
        card_visible = false;
      }
      auto getvisiblename = [&](auto& p) {
        return card_visible ? card.name_ : "Face-down card";
      };

      if ((reason & REASON_DESTROY) && (card.location_ != cnew.location_)) {
        pl->notify(fmt::format("Card {} ({}) destroyed.", plspec, card.name_));
        op->notify(fmt::format("Card {} ({}) destroyed.", opspec, card.name_));
      } else if ((card.location_ == cnew.location_) &&
                 (card.location_ & LOCATION_ONFIELD)) {
        if (card.controler_ != cnew.controler_) {
          pl->notify(
              fmt::format("Your card {} ({}) changed controller to {} and is "
                          "now located at {}.",
                          plspec, card.name_, op->nickname_, plnewspec));
          op->notify(
              fmt::format("You now control {}'s card {} ({}) and it's located "
                          "at {}.",
                          pl->nickname_, opspec, card.name_, opnewspec));
        } else {
          pl->notify(fmt::format("Your card {} ({}) switched its zone to {}.",
                                 plspec, card.name_, plnewspec));
          op->notify(fmt::format("{}'s card {} ({}) switched its zone to {}.",
                                 pl->nickname_, opspec, card.name_, opnewspec));
        }
      } else if ((reason & REASON_DISCARD) &&
                 (card.location_ != cnew.location_)) {
        pl->notify(fmt::format("You discarded {} ({})", plspec, card.name_));
        op->notify(fmt::format("{} discarded {} ({})", pl->nickname_, opspec,
                               card.name_));
      } else if ((card.location_ == LOCATION_REMOVED) &&
                 (cnew.location_ & LOCATION_ONFIELD)) {
        pl->notify(
            fmt::format("Your banished card {} ({}) returns to the field at "
                        "{}.",
                        plspec, card.name_, plnewspec));
        op->notify(
            fmt::format("{}'s banished card {} ({}) returns to the field at "
                        "{}.",
                        pl->nickname_, opspec, card.name_, opnewspec));
      } else if ((card.location_ == LOCATION_GRAVE) &&
                 (cnew.location_ & LOCATION_ONFIELD)) {
        pl->notify(
            fmt::format("Your card {} ({}) returns from the graveyard to the "
                        "field at {}.",
                        plspec, card.name_, plnewspec));
        op->notify(
            fmt::format("{}'s card {} ({}) returns from the graveyard to the "
                        "field at {}.",
                        pl->nickname_, opspec, card.name_, opnewspec));
      } else if ((cnew.location_ == LOCATION_HAND) &&
                 (card.location_ != cnew.location_)) {
        pl->notify(
            fmt::format("Card {} ({}) returned to hand.", plspec, card.name_));
      } else if ((reason & (REASON_RELEASE | REASON_SUMMON)) &&
                 (card.location_ != cnew.location_)) {
        pl->notify(fmt::format("You tribute {} ({}).", plspec, card.name_));
        op->notify(fmt::format("{} tributes {} ({}).", pl->nickname_, opspec,
                               getvisiblename(op)));
      } else if ((card.location_ == (LOCATION_OVERLAY | LOCATION_MZONE)) &&
                 (cnew.location_ & LOCATION_GRAVE)) {
        pl->notify(fmt::format("You detached {}.", card.name_));
        op->notify(fmt::format("{} detached {}.", pl->nickname_, card.name_));
      } else if ((card.location_ != cnew.location_) &&
                 (cnew.location_ == LOCATION_GRAVE)) {
        pl->notify(fmt::format("Your card {} ({}) was sent to the graveyard.",
                               plspec, card.name_));
        op->notify(fmt::format("{}'s card {} ({}) was sent to the graveyard.",
                               pl->nickname_, opspec, card.name_));
      } else if ((card.location_ != cnew.location_) &&
                 (cnew.location_ == LOCATION_REMOVED)) {
        pl->notify(
            fmt::format("Your card {} ({}) was banished.", plspec, card.name_));
        op->notify(fmt::format("{}'s card {} ({}) was banished.", pl->nickname_,
                               opspec, getvisiblename(op)));
      } else if ((card.location_ != cnew.location_) &&
                 (cnew.location_ == LOCATION_DECK)) {
        pl->notify(fmt::format("Your card {} ({}) returned to your deck.",
                               plspec, card.name_));
        op->notify(fmt::format("{}'s card {} ({}) returned to their deck.",
                               pl->nickname_, opspec, getvisiblename(op)));
      } else if ((card.location_ != cnew.location_) &&
                 (cnew.location_ == LOCATION_EXTRA)) {
        pl->notify(fmt::format("Your card {} ({}) returned to your extra deck.",
                               plspec, card.name_));
        op->notify(
            fmt::format("{}'s card {} ({}) returned to their extra deck.",
                        pl->nickname_, opspec, getvisiblename(op)));
      } else if ((card.location_ == LOCATION_DECK) &&
                 (cnew.location_ == LOCATION_SZONE) &&
                 (cnew.position_ != POS_FACEDOWN)) {
        pl->notify(fmt::format("Activating {} ({})", plnewspec, card.name_));
        op->notify(fmt::format("{} activating {} ({})", pl->nickname_, opspec,
                               cnew.name_));
      }
    } else if (msg_ == MSG_SWAP) {
      CardCode code1 = read_u32();
      uint32_t loc1 = read_u32();
      CardCode code2 = read_u32();
      uint32_t loc2 = read_u32();
      if (!verbose_) return;
      Card cards[2];
      cards[0] = c_get_card(code1);
      cards[1] = c_get_card(code2);
      cards[0].set_location(loc1);
      cards[1].set_location(loc2);

      for (PlayerId pl = 0; pl < 2; pl++) {
        for (int i = 0; i < 2; i++) {
          auto c = cards[i];
          auto spec = c.get_spec(pl);
          auto plname = players_[1 - c.controler_]->nickname_;
          players_[pl]->notify("Card " + c.name_ + " swapped control towards " +
                               plname + " and is now located at " + spec + ".");
        }
      }
    } else if (msg_ == MSG_SET) {
      CardCode code = read_u32();
      uint32_t location = read_u32();
      if (!verbose_) return;
      Card card = c_get_card(code);
      card.set_location(location);
      auto c = card.controler_;
      auto& cpl = players_[c];
      auto& opl = players_[1 - c];
      cpl->notify(fmt::format("You set {} ({}) in {} position.", card.name_,
                              card.get_spec(c), card.get_position()));
      opl->notify(fmt::format("{} sets {} in {} position.", cpl->nickname_,
                              card.get_spec(PlayerId(1 - c)),
                              card.get_position()));
    } else if (msg_ == MSG_EQUIP) {
      auto c = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      auto pos = read_u8();
      auto tc = read_u8();
      auto tloc = read_u8();
      auto tseq = read_u8();
      auto tpos = read_u8();
      if (!verbose_) return;
      Card card = get_card(c, loc, seq);
      Card target = get_card(tc, tloc, tseq);
      for (PlayerId pl = 0; pl < 2; pl++) {
        auto c = cardlist_info_for_player(card, pl);
        auto t = cardlist_info_for_player(target, pl);
        players_[pl]->notify(fmt::format("{} equipped to {}.", c, t));
      }
    } else if (msg_ == MSG_HINT) {
      auto hint_type = read_u8();
      auto player = read_u8();
      auto value = read_u32();

      if (hint_type == HINT_SELECTMSG && value == 501) {
        discard_hand_ = true;
      }
      // non-GUI don't need hint
      return;
      if (hint_type == HINT_SELECTMSG) {
        if (value > 2000) {
          CardCode code = value;
          players_[player]->notify(fmt::format("{} select {}",
                                               players_[player]->nickname_,
                                               c_get_card(code).name_));
        } else {
          players_[player]->notify(get_system_string(value));
        }
      } else if (hint_type == HINT_NUMBER) {
        players_[1 - player]->notify(
            fmt::format("Choice of player: {}", value));
      } else {
        fmt::println("Unknown hint type {} with value {}", hint_type, value);
      }
    } else if (msg_ == MSG_CARD_HINT) {
      uint8_t player = read_u8();
      uint8_t loc = read_u8();
      uint8_t seq = read_u8();
      uint8_t pos = read_u8();
      uint8_t type = read_u8();
      uint32_t value = read_u32();
      if (!verbose_) return;
      if (type == CHINT_RACE) {
        Card card = get_card(player, loc, seq);
        if (card.code_ == 0) {
          return;
        }
        std::string races_str = "TODO";
        for (PlayerId pl = 0; pl < 2; pl++) {
          players_[pl]->notify(fmt::format("{} ({}) selected {}.",
                                           card.get_spec(pl), card.name_,
                                           races_str));
        }
      } else if (type == CHINT_ATTRIBUTE) {
        Card card = get_card(player, loc, seq);
        if (card.code_ == 0) {
          return;
        }
        std::string attributes_str = "TODO";
        for (PlayerId pl = 0; pl < 2; pl++) {
          players_[pl]->notify(fmt::format("{} ({}) selected {}.",
                                           card.get_spec(pl), card.name_,
                                           attributes_str));
        }
      } else {
        fmt::println("Unknown card hint type {} with value {}", type, value);
      }
    } else if (msg_ == MSG_POS_CHANGE) {
      CardCode code = read_u32();
      Card card = c_get_card(code);
      card.set_location(read_u32());
      uint8_t prevpos = card.position_;
      card.position_ = read_u8();
      if (!verbose_) return;

      auto& pl = players_[card.controler_];
      auto& op = players_[1 - card.controler_];
      auto plspec = card.get_spec(false);
      auto opspec = card.get_spec(true);
      auto prevpos_str = position_to_string(prevpos);
      auto pos_str = position_to_string(card.position_);
      pl->notify("The position of card " + plspec + " (" + card.name_ +
                 ") changed from " + prevpos_str + " to " + pos_str + ".");
      op->notify("The position of card " + opspec + " (" + card.name_ +
                 ") changed from " + prevpos_str + " to " + pos_str + ".");
    } else if (msg_ == MSG_BECOME_TARGET) {
      auto u = read_u8();
      uint32_t target = read_u32();
      uint8_t tc = target & 0xff;
      uint8_t tl = (target >> 8) & 0xff;
      uint8_t tseq = (target >> 16) & 0xff;
      Card card = get_card(tc, tl, tseq);
      add_public_event(kEventTarget, chaining_player_, visible_ref(card));
      if (!verbose_) return;
      auto name = players_[chaining_player_]->nickname_;
      for (PlayerId pl = 0; pl < 2; pl++) {
        auto spec = card.get_spec(pl);
        auto tcname = card.name_;
        if ((card.controler_ != pl) && (card.position_ & POS_FACEDOWN)) {
          tcname = position_to_string(card.position_) + " card";
        }
        players_[pl]->notify(name + " targets " + spec + " (" + tcname + ")");
      }
    } else if (msg_ == MSG_CONFIRM_DECKTOP ||
               msg_ == MSG_CONFIRM_EXTRATOP) {
      auto player = read_u8();
      auto size = read_u8();
      std::vector<Card> cards;
      for (int i = 0; i < size; ++i) {
        read_u32();
        auto c = read_u8();
        auto loc = read_u8();
        auto seq = read_u8();
        revealed_.insert(ls_to_spec(loc, seq, 0, c != player));
        if (verbose_) cards.push_back(get_card(c, loc, seq));
      }
      if (!verbose_) return;

      for (PlayerId pl = 0; pl < 2; pl++) {
        auto& p = players_[pl];
        const auto zone = msg_ == MSG_CONFIRM_EXTRATOP ? "extra deck" : "deck";
        if (pl == player) {
          p->notify(fmt::format("You reveal {} cards from your {}:", size, zone));
        } else {
          p->notify(fmt::format("{} reveals {} cards from their {}:",
                                players_[player]->nickname_, size, zone));
        }
        for (int i = 0; i < size; ++i) {
          p->notify(fmt::format("{}: {}", i + 1, cards[i].name_));
        }
      }
    } else if (msg_ == MSG_RANDOM_SELECTED) {
      auto player = read_u8();
      auto count = read_u8();
      std::vector<Card> cards;

      for (int i = 0; i < count; ++i) {
        auto c = read_u8();
        auto loc = read_u8();
        if (loc & LOCATION_OVERLAY) {
          throw std::runtime_error("Overlay not supported for random selected");
        }
        auto seq = read_u8();
        auto pos = read_u8();
        if (verbose_) cards.push_back(get_card(c, loc, seq));
      }
      if (!verbose_) return;

      for (PlayerId pl = 0; pl < 2; pl++) {
        auto& p = players_[pl];
        auto s = "card is";
        if (count > 1) {
          s = "cards are";
        }
        if (pl == player) {
          p->notify(fmt::format("Your {} {} randomly selected:", s, count));
        } else {
          p->notify(fmt::format("{}'s {} {} randomly selected:",
                                players_[player]->nickname_, s, count));
        }
        for (int i = 0; i < count; ++i) {
          p->notify(fmt::format("{}: {}", cards[i].get_spec(pl), cards[i].name_));
        }
      }
    } else if (msg_ == MSG_TOSS_COIN || msg_ == MSG_TOSS_DICE) {
      // Informational random-result messages.  They do not require a player
      // response, but they must still be consumed so the message stream stays
      // aligned.  The payload is player, result count, then one byte per
      // result (coin: 0/1, die: 1..6).
      auto player = read_u8();
      auto count = read_u8();
      std::vector<uint8_t> results;
      results.reserve(count);
      for (int i = 0; i < count; ++i) {
        results.push_back(read_u8());
      }
      if (verbose_) {
        std::string values;
        for (int i = 0; i < results.size(); ++i) {
          if (i != 0) {
            values += ", ";
          }
          if (msg_ == MSG_TOSS_COIN) {
            values += results[i] == 0 ? "tails" : "heads";
          } else {
            values += std::to_string(results[i]);
          }
        }
        players_[player]->notify(fmt::format(
          "{} result(s): {}",
          msg_ == MSG_TOSS_COIN ? "Coin toss" : "Dice roll", values));
      }
    } else if (msg_ == MSG_FIELD_DISABLED) {
      // Informational 32-bit zone mask; no response is required.
      auto disabled = read_u32();
      if (verbose_) {
        players_[0]->notify(fmt::format("Disabled field mask: 0x{:08x}", disabled));
        players_[1]->notify(fmt::format("Disabled field mask: 0x{:08x}", disabled));
      }
    } else if (msg_ == MSG_DECK_TOP) {
      // Informational reveal/update for a card near the top of a deck.
      auto player = read_u8();
      auto sequence = read_u8();
      auto code = read_u32();
      if (verbose_) {
        players_[player]->notify(fmt::format(
          "Deck-top update at {}: card {}", sequence, code & 0x7fffffff));
      }
    } else if (msg_ == MSG_PLAYER_HINT) {
      skip_message_bytes(6);
      // TODO(3): implement output
    } else if (msg_ == MSG_CARD_TARGET) {
      auto c1 = read_u8();
      auto l1 = read_u8();
      auto s1 = read_u8();
      read_u8();
      auto c2 = read_u8();
      auto l2 = read_u8();
      auto s2 = read_u8();
      read_u8();
      if (!verbose_) return;

      Card card1 = get_card(c1, l1, s1);
      Card card2 = get_card(c2, l2, s2);
      for (PlayerId pl = 0; pl < 2; pl++) {
        auto& p = players_[pl];
        auto spec1 = card1.get_spec(pl);
        auto spec2 = card2.get_spec(pl);
        auto c1name = card1.name_;
        auto c2name = card2.name_;
        if ((card1.controler_ != pl) && (card1.position_ & POS_FACEDOWN)) {
          c1name = position_to_string(card1.position_) + " card";
        }
        if ((card2.controler_ != pl) && (card2.position_ & POS_FACEDOWN)) {
          c2name = position_to_string(card2.position_) + " card";
        }
        p->notify(fmt::format(" {} ({}) targets {} ({})", spec1, c1name, spec2, c2name));
      }
    } else if (msg_ == MSG_CANCEL_TARGET) {
      const uint32_t source = read_u32();
      const uint32_t target = read_u32();
      if (verbose_) {
        for (auto &pl : players_) {
          pl->notify(fmt::format(
              "Card target relation removed: 0x{:08x} -> 0x{:08x}",
              source, target));
        }
      }
    } else if (msg_ == MSG_CONFIRM_CARDS) {
      auto player = read_u8();
      auto size = read_u8();
      std::vector<Card> cards;
      for (int i = 0; i < size; ++i) {
        read_u32();
        auto c = read_u8();
        auto loc = read_u8();
        auto seq = read_u8();
        if (verbose_) {
          cards.push_back(get_card(c, loc, seq));
        }
        revealed_.insert(ls_to_spec(loc, seq, 0, c == player));
      }
      if (!verbose_) {
        return;
      }

      auto& pl = players_[player];
      auto& op = players_[1 - player];

      op->notify(fmt::format("{} shows you {} cards.", pl->nickname_, size));
      for (int i = 0; i < size; ++i) {
        pl->notify(fmt::format("{}: {}", i + 1, cards[i].name_));
      }
    } else if (msg_ == MSG_MISSED_EFFECT) {
      skip_message_bytes(4);
      CardCode code = read_u32();
      if (!verbose_) return;
      Card card = c_get_card(code);
      for (PlayerId pl = 0; pl < 2; pl++) {
        auto spec = card.get_spec(pl);
        auto str = get_system_string(1622);
        std::string fmt_str = "[%ls]";
        str = str.replace(str.find(fmt_str), fmt_str.length(), card.name_);
        players_[pl]->notify(str);
      }
    } else if (msg_ == MSG_SORT_CARD) {
      auto player = read_u8();
      auto size = read_u8();
      n_sort_cards_ = size;
      std::vector<Card> cards;
      std::vector<std::string> specs;
      specs.reserve(size);
      for (int i = 0; i < size; ++i) {
        read_u32();
        auto c = read_u8();
        auto loc = read_u8();
        auto seq = read_u8();
        specs.push_back(ls_to_spec(loc, seq, 0, c != player));
        if (verbose_) cards.push_back(get_card(c, loc, seq));
      }
      if (verbose_) {
        auto& pl = players_[player];
        pl->notify("Keeping the current order for " + std::to_string(size) +
                   " sortable cards.");
        for (int i = 0; i < size; ++i) {
          pl->notify(fmt::format("{}: {}", i + 1, cards[i].name_));
        }
      }
      // The no-reorder sentinel and one-card-at-a-time ordering choices
      // expose O(size) actions per step, never size! permutations at once.
      init_multi_select(size, size, 0, specs, 4);
      to_play_ = player;
      callback_ = [this](int idx) { callback_sort_selection(idx); };
      return;
    } else if (msg_ == MSG_ROCK_PAPER_SCISSORS) {
      // ygopro-core asks each player for an integer hand in [1, 3]. The
      // trailing byte identifies which player's choice is currently due.
      auto player = core_rock_paper_scissors_player(read_u8());
      for (uint32_t hand : core_rock_paper_scissors_options()) {
        legal_actions_.push_back(LegalAction::number(hand));
      }
      to_play_ = player;
      callback_ = [this](int idx) {
        if (idx < 0 || idx >= static_cast<int>(legal_actions_.size())) {
          throw std::runtime_error("Invalid rock-paper-scissors action index");
        }
        const uint32_t hand = legal_actions_[idx].number_;
        core_scalar_response(static_cast<int32_t>(hand), 1, 3);
        YGO_SetResponsei(pduel_, static_cast<int32_t>(hand));
      };
      return;
    } else if (msg_ == MSG_HAND_RES) {
      const auto packed = read_u8();
      const auto first = packed & 0x3;
      const auto second = (packed >> 2) & 0x3;
      if (first < 1 || first > 3 || second < 1 || second > 3 ||
          (packed & 0xf0) != 0) {
        throw std::runtime_error("Invalid MSG_HAND_RES choices");
      }
      if (verbose_) {
        for (auto &pl : players_) {
          pl->notify(fmt::format(
              "Rock-paper-scissors result: player 0 chose {}, player 1 chose {}.",
              first, second));
        }
      }
    } else if (msg_ == MSG_ADD_COUNTER) {
      auto ctype = read_u16();
      auto player = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      auto count = read_u16();
      if (!verbose_) return;
      auto c = get_card(player, loc, seq);
      auto& pl = players_[player];
      PlayerId op_id = 1 - player;
      auto& op = players_[op_id];
      // TODO(3): counter type to string
      pl->notify(fmt::format("{} counter(s) of type {} placed on {} ().", count, "UNK", c.name_, c.get_spec(player)));
      op->notify(fmt::format("{} counter(s) of type {} placed on {} ().", count, "UNK", c.name_, c.get_spec(op_id)));
    } else if (msg_ == MSG_REMOVE_COUNTER) {
      auto ctype = read_u16();
      auto player = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      auto count = read_u16();
      if (!verbose_) return;
      auto c = get_card(player, loc, seq);
      auto& pl = players_[player];
      PlayerId op_id = 1 - player;
      auto& op = players_[op_id];
      pl->notify(fmt::format("{} counter(s) of type {} removed from {} ().", count, "UNK", c.name_, c.get_spec(player)));
      op->notify(fmt::format("{} counter(s) of type {} removed from {} ().", count, "UNK", c.name_, c.get_spec(op_id)));
    } else if (msg_ == MSG_ATTACK_DISABLED) {
      if (verbose_) {
        for (PlayerId pl = 0; pl < 2; pl++) {
          players_[pl]->notify(get_system_string(1621));
        }
      }
    } else if (msg_ == MSG_SHUFFLE_SET_CARD) {
      const auto location = read_u8();
      const auto count = read_u8();
      for (int i = 0; i < count; ++i) read_u32();
      for (int i = 0; i < count; ++i) read_u32();
      if (verbose_) {
        for (auto &pl : players_) {
          pl->notify(fmt::format(
              "Shuffled {} facedown set card(s) in location {}.",
              count, location));
        }
      }
    } else if (msg_ == MSG_SHUFFLE_DECK) {
      auto player = read_u8();
      if (player > 1) {
        throw std::runtime_error("Invalid MSG_SHUFFLE_DECK player");
      }
      if (!verbose_) return;
      auto& pl = players_[player];
      auto& op = players_[1 - player];
      pl->notify("You shuffled your deck.");
      op->notify(pl->nickname_ + " shuffled their deck.");
    } else if (msg_ == MSG_SWAP_GRAVE_DECK) {
      const auto player = read_u8();
      if (player > 1) {
        throw std::runtime_error("Invalid MSG_SWAP_GRAVE_DECK player");
      }
      revealed_.clear();
      if (verbose_) {
        players_[player]->notify("Your deck and graveyard were exchanged.");
        players_[1 - player]->notify(fmt::format(
            "{} exchanged their deck and graveyard.",
            players_[player]->nickname_));
      }
    } else if (msg_ == MSG_REVERSE_DECK) {
      // Notification-only message with no payload.  Deck contents and order
      // are queried from the core when observations are built; consuming the
      // message byte is therefore sufficient.  Do not advance to dl_ here:
      // the core can concatenate DECK_TOP and other notifications in the same
      // buffer immediately after this one.
      if (verbose_) {
        for (auto &pl : players_) {
          pl->notify("The deck order was reversed.");
        }
      }
    } else if (msg_ == MSG_SHUFFLE_EXTRA) {
      auto player = read_u8();
      auto count = read_u8();
      for (int i = 0; i < count; ++i) {
        read_u32();
      }
      if (!verbose_) return;
      auto& pl = players_[player];
      auto& op = players_[1 - player];
      pl->notify(fmt::format("You shuffled your extra deck ({}).", count));
      op->notify(fmt::format("{} shuffled their extra deck ({}).", pl->nickname_, count));
    } else if (msg_ == MSG_SHUFFLE_HAND) {
      auto player = read_u8();
      auto count = read_u8();
      for (int i = 0; i < count; ++i) read_u32();
      if (!verbose_) return;
      auto& pl = players_[player];
      auto& op = players_[1 - player];
      pl->notify("You shuffled your hand.");
      op->notify(pl->nickname_ + " shuffled their hand.");
    } else if (msg_ == MSG_TAG_SWAP) {
      const auto player = read_u8();
      const auto main_count = read_u8();
      const auto extra_count = read_u8();
      const auto extra_faceup_count = read_u8();
      const auto hand_count = read_u8();
      read_u32();
      for (int i = 0; i < hand_count; ++i) read_u32();
      for (int i = 0; i < extra_count; ++i) read_u32();
      if (player > 1 || extra_faceup_count > extra_count) {
        throw std::runtime_error("Invalid MSG_TAG_SWAP payload");
      }
      revealed_.clear();
      if (verbose_) {
        for (auto &pl : players_) {
          pl->notify(fmt::format(
              "Tag player {} swapped in (deck {}, hand {}, extra {}/{} face-up).",
              player, main_count, hand_count, extra_count,
              extra_faceup_count));
        }
      }
    } else if (msg_ == MSG_SUMMONED) {
      // Notification-only marker with no payload.
    } else if (msg_ == MSG_SUMMONING) {
      CardCode code = read_u32();
      Card card = c_get_card(code);
      card.set_location(read_u32());
      add_public_event(kEventSummon, card.controler_, visible_ref(card));
      if (!verbose_) return;
      const auto &nickname = players_[card.controler_]->nickname_;
      for (auto& pl : players_) {
        pl->notify(nickname + " summoning " + card.name_ + " (" +
                   std::to_string(card.attack_) + "/" +
                   std::to_string(card.defense_) + ") in " +
                   card.get_position() + " position.");
      }
    } else if (msg_ == MSG_SPSUMMONED) {
      // Notification-only marker with no payload.
    } else if (msg_ == MSG_FLIPSUMMONED) {
      // Notification-only marker with no payload.
    } else if (msg_ == MSG_FLIPSUMMONING) {
      auto code = read_u32();
      auto location = read_u32();
      Card card = c_get_card(code);
      card.set_location(location);
      add_public_event(kEventSummon, card.controler_, visible_ref(card));
      if (!verbose_) return;

      auto& cpl = players_[card.controler_];
      for (PlayerId pl = 0; pl < 2; pl++) {
        auto spec = card.get_spec(pl);
        players_[1 - pl]->notify(cpl->nickname_ + " flip summons " + spec +
                                 " (" + card.name_ + ")");
      }
    } else if (msg_ == MSG_SPSUMMONING) {
      CardCode code = read_u32();
      Card card = c_get_card(code);
      card.set_location(read_u32());
      add_public_event(kEventSpecialSummon, card.controler_, visible_ref(card));
      if (!verbose_) return;
      const auto &nickname = players_[card.controler_]->nickname_;
      for (PlayerId p = 0; p < 2; p++) {
        auto& pl = players_[p];
        auto pos = card.get_position();
        auto atk = std::to_string(card.attack_);
        auto def = std::to_string(card.defense_);
        std::string name = p == card.controler_ ? "You" : nickname;
        if (card.type_ & TYPE_LINK) {
          pl->notify(name + " special summoning " + card.name_ + " (" +
                     atk + ") in " + pos + " position.");
        } else {
          pl->notify(name + " special summoning " + card.name_ + " (" +
                     atk + "/" + def + ") in " + pos + " position.");
        }
      }
    } else if (msg_ == MSG_CHAIN_NEGATED) {
      read_u8();
      add_public_event(kEventNegation, chaining_player_, active_chain_source_, 0, 0, 1);
    } else if (msg_ == MSG_CHAIN_DISABLED) {
      read_u8();
      add_public_event(kEventNegation, chaining_player_, active_chain_source_, 0, 0, 2);
    } else if (msg_ == MSG_CHAIN_SOLVED) {
      read_u8();
      add_public_event(kEventChainSolved, chaining_player_, active_chain_source_);
      chain_depth_ = std::max(0, chain_depth_ - 1);
      revealed_.clear();
    } else if (msg_ == MSG_CHAIN_SOLVING) {
      read_u8();
      add_public_event(kEventChainSolving, chaining_player_, active_chain_source_);
    } else if (msg_ == MSG_CHAINED) {
      read_u8();
    } else if (msg_ == MSG_CHAIN_END) {
      add_public_event(kEventChainEnd, chaining_player_, active_chain_source_);
      chain_depth_ = 0;
      active_chain_source_ = VisibleCardRef{};
    } else if (msg_ == MSG_CHAINING) {
      CardCode code = read_u32();
      Card card = c_get_card(code);
      card.set_location(read_u32());
      auto tc = read_u8();
      auto tl = read_u8();
      auto ts = read_u8();
      uint32_t desc = read_u32();
      auto cs = read_u8();
      auto c = card.controler_;
      PlayerId o = 1 - c;
      chaining_player_ = c;
      chain_depth_ = std::max(chain_depth_, static_cast<int>(cs));
      active_chain_source_ = visible_ref(card);
      add_public_event(kEventActivation, c, active_chain_source_);
      if (!verbose_) return;
      players_[c]->notify("Activating " + card.get_spec(c) + " (" + card.name_ +
                          ")");
      players_[o]->notify(players_[c]->nickname_ + " activating " +
                          card.get_spec(o) + " (" + card.name_ + ")");
    } else if (msg_ == MSG_DAMAGE) {
      auto player = read_u8();
      auto amount = read_u32();
      _damage(player, amount);
    } else if (msg_ == MSG_RECOVER) {
      auto player = read_u8();
      auto amount = read_u32();
      _recover(player, amount);
    } else if (msg_ == MSG_LPUPDATE) {
      auto player = read_u8();
      auto lp = read_u32();
      if (lp >= lp_[player]) {
        _recover(player, lp - lp_[player]);
      } else {
        _damage(player, lp_[player] - lp);
      }
    } else if (msg_ == MSG_PAY_LPCOST) {
      auto player = read_u8();
      auto cost = read_u32();
      lp_[player] -= cost;
      if (!verbose_) {
        return;
      }
      auto& pl = players_[player];
      pl->notify("You pay " + std::to_string(cost) + " LP. Your LP is now " +
                 std::to_string(lp_[player]) + ".");
      players_[1 - player]->notify(
          pl->nickname_ + " pays " + std::to_string(cost) + " LP. " +
          pl->nickname_ + "'s LP is now " + std::to_string(lp_[player]) + ".");
    } else if (msg_ == MSG_ATTACK) {
      auto attacker = read_u32();
      PlayerId ac = attacker & 0xff;
      auto aloc = (attacker >> 8) & 0xff;
      auto aseq = (attacker >> 16) & 0xff;
      auto apos = (attacker >> 24) & 0xff;
      auto target = read_u32();
      PlayerId tc = target & 0xff;
      auto tloc = (target >> 8) & 0xff;
      auto tseq = (target >> 16) & 0xff;
      auto tpos = (target >> 24) & 0xff;
      if (!verbose_) return;

      if ((ac == 0) && (aloc == 0) && (aseq == 0) && (apos == 0)) {
        return;
      }

      Card acard = get_card(ac, aloc, aseq);
      auto name = players_[ac]->nickname_;
      if ((tc == 0) && (tloc == 0) && (tseq == 0) && (tpos == 0)) {
        for (PlayerId i = 0; i < 2; i++) {
          players_[i]->notify(name + " prepares to attack with " +
                              acard.get_spec(i) + " (" + acard.name_ + ")");
        }
        return;
      }

      Card tcard = get_card(tc, tloc, tseq);
      for (PlayerId i = 0; i < 2; i++) {
        auto aspec = acard.get_spec(i);
        auto tspec = tcard.get_spec(i);
        auto tcname = tcard.name_;
        if ((tcard.controler_ != i) && (tcard.position_ & POS_FACEDOWN)) {
          tcname = tcard.get_position() + " card";
        }
        players_[i]->notify(name + " prepares to attack " + tspec + " (" +
                            tcname + ") with " + aspec + " (" + acard.name_ +
                            ")");
      }
    } else if (msg_ == MSG_DAMAGE_STEP_START) {
      if (!verbose_) {
        return;
      }
      for (int i = 0; i < 2; i++) {
        players_[i]->notify("begin damage");
      }
    } else if (msg_ == MSG_DAMAGE_STEP_END) {
      if (!verbose_) {
        return;
      }
      for (int i = 0; i < 2; i++) {
        players_[i]->notify("end damage");
      }
    } else if (msg_ == MSG_BATTLE) {
      auto attacker = read_u32();
      auto aa = read_u32();
      auto ad = read_u32();
      auto bd0 = read_u8();
      auto target = read_u32();
      auto da = read_u32();
      auto dd = read_u32();
      auto bd1 = read_u8();
      if (!verbose_) return;

      auto ac = attacker & 0xff;
      auto aloc = (attacker >> 8) & 0xff;
      auto aseq = (attacker >> 16) & 0xff;

      auto tc = target & 0xff;
      auto tloc = (target >> 8) & 0xff;
      auto tseq = (target >> 16) & 0xff;
      auto tpos = (target >> 24) & 0xff;

      Card acard = get_card(ac, aloc, aseq);
      Card tcard;
      if (tloc != 0) {
        tcard = get_card(tc, tloc, tseq);
      }
      for (int i = 0; i < 2; i++) {
        auto& pl = players_[i];
        std::string attacker_points;
        if (acard.type_ & TYPE_LINK) {
          attacker_points = std::to_string(aa);
        } else {
          attacker_points = std::to_string(aa) + "/" + std::to_string(ad);
        }
        if (tloc != 0) {
          std::string defender_points;
          if (tcard.type_ & TYPE_LINK) {
            defender_points = std::to_string(da);
          } else {
            defender_points = std::to_string(da) + "/" + std::to_string(dd);
          }
          pl->notify(acard.name_ + "(" + attacker_points + ")" + " attacks " +
                     tcard.name_ + " (" + defender_points + ")");
        } else {
          pl->notify(acard.name_ + "(" + attacker_points + ")" + " attacks");
        }
      }
    } else if (msg_ == MSG_MATCH_KILL) {
      const CardCode code = read_u32();
      if (verbose_) {
        for (auto &pl : players_) {
          pl->notify(fmt::format(
              "Match-winning effect applied by card {}.", code));
        }
      }
    } else if (msg_ == MSG_WIN) {
      auto player = read_u8();
      auto reason = read_u8();
      auto& winner = players_[player];
      auto& loser = players_[1 - player];

      _duel_end(player, reason);

      auto l_reason = reason_to_string(reason);
      if (verbose_) {
        winner->notify("You won (" + l_reason + ").");
        loser->notify("You lost (" + l_reason + ").");
      }
    } else if (msg_ == MSG_RETRY) {
      throw std::runtime_error("Retry");
    } else if (msg_ == MSG_SELECT_BATTLECMD) {
      auto player = read_u8();
      auto activatable = read_cardlist_spec(player, true);
      auto attackable = read_cardlist_spec(player, true, true);
      bool to_m2 = read_u8();
      bool to_ep = read_u8();

      auto& pl = players_[player];
      if (verbose_) {
        pl->notify("Battle menu:");
      }
      for (const auto [code_t, spec, desc] : activatable) {
        CardCode code = code_t;
        if(code & 0x80000000) {
          code &= 0x7fffffff;
        }
        auto [code_d, eff_idx] = unpack_desc(code, desc);
        if (desc == 0) {
          code_d = code;
        }
        auto la = LegalAction::activate_spec(eff_idx, spec);
        if (code_d != 0) {
          la.cid_ = c_get_card_id(code_d);
        }
        legal_actions_.push_back(la);
        if (verbose_) {
          auto c = c_get_card(code);
          int cmd_idx = legal_actions_.size();
          std::string s = fmt::format(
            "{}: activate {}({}) [{}/{}] ({})",
            cmd_idx, c.name_, spec, c.attack_, c.defense_, c.get_effect_description(code_d, eff_idx));
        }
      }
      for (const auto [code, spec, data] : attackable) {
        bool direct_attackable = data & 0x1;
        auto act = direct_attackable ? ActionAct::DirectAttack : ActionAct::Attack;

        legal_actions_.push_back(
          LegalAction::act_spec(act, spec));
        if (verbose_) {
          auto [controller, loc, seq, pos] = spec_to_ls(player, spec);
          auto c = get_card(controller, loc, seq);
          int cmd_idx = legal_actions_.size();
          auto attack_str = direct_attackable ? "direct attack" : "attack";
          std::string s = fmt::format(
            "{}: {} {}({}) ", cmd_idx, attack_str, c.name_, spec);
          if (c.type_ & TYPE_LINK) {
            s += fmt::format("[{}]", c.attack_);
          } else {
            s += fmt::format("[{}/{}]", c.attack_, c.defense_);
          }
          pl->notify(s);
        }
      }
      if (to_m2) {
        legal_actions_.push_back(
          LegalAction::phase(ActionPhase::Main2));
        int cmd_idx = legal_actions_.size();
        if (verbose_) {
          pl->notify(fmt::format("{}: Main phase 2.", cmd_idx));
        }
      }
      if (to_ep) {
        legal_actions_.push_back(
          LegalAction::phase(ActionPhase::End));
        int cmd_idx = legal_actions_.size();
        if (verbose_) {
          pl->notify(fmt::format("{}: End phase.", cmd_idx));
        }
      }
      int n_activatables = activatable.size();
      int n_attackables = attackable.size();
      to_play_ = player;
      callback_ = [this, n_activatables, n_attackables, to_ep, to_m2](int idx) {
        const auto &la = legal_actions_[idx];
        if (idx < n_activatables) {
          YGO_SetResponsei(pduel_, idx << 16);
        } else if (idx < (n_activatables + n_attackables)) {
          idx = idx - n_activatables;
          YGO_SetResponsei(pduel_, (idx << 16) + 1);
        } else if ((la.phase_ == ActionPhase::End) && to_ep) {
          YGO_SetResponsei(pduel_, 3);
        } else if ((la.phase_ == ActionPhase::Main2) && to_m2) {
          YGO_SetResponsei(pduel_, 2);
        } else {
          throw std::runtime_error("Invalid option");
        }
      };
    } else if (msg_ == MSG_SELECT_UNSELECT_CARD) {
      // TODO: add feature of selected cards (also for multi select)
      auto player = read_u8();
      bool finishable = read_u8();
      bool cancelable = read_u8();
      auto min = read_u8();
      auto max = read_u8();
      auto select_size = read_u8();
      selection_finishable_ = finishable;
      selection_cancelable_ = cancelable;

      std::vector<std::string> select_specs;
      select_specs.reserve(select_size);
      if (verbose_) {
        auto& pl = players_[player];
        pl->notify("Select " + std::to_string(min) + " to " +
                   std::to_string(max) + " cards:");
        for (int i = 0; i < select_size; ++i) {
          auto code = read_u32();
          auto loc = read_u32();
          Card card = c_get_card(code);
          card.set_location(loc);
          auto spec = card.get_spec(player);
          select_specs.push_back(spec);
          auto s = fmt::format("{}: {}({})", i + 1, card.name_, spec);
          pl->notify(s);
        }
      } else {
        for (int i = 0; i < select_size; ++i) {
          dp_ += 4;
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto pos = read_u8();
          auto spec = ls_to_spec(loc, seq, pos, controller != player);
          select_specs.push_back(spec);
        }
      }

      auto unselect_size = read_u8();
      std::vector<std::string> unselect_specs;
      unselect_specs.reserve(unselect_size);
      if (verbose_) {
        auto& pl = players_[player];
        for (int i = 0; i < unselect_size; ++i) {
          auto code = read_u32();
          auto loc = read_u32();
          Card card = c_get_card(code);
          card.set_location(loc);
          auto spec = card.get_spec(player);
          unselect_specs.push_back(spec);
          pl->notify(fmt::format("{}: unselect {} ({})",
                                 select_specs.size() + i + 1, card.name_, spec));
        }
      } else {
        for (int i = 0; i < unselect_size; ++i) {
          dp_ += 4;
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto pos = read_u8();
          unselect_specs.push_back(
              ls_to_spec(loc, seq, pos, controller != player));
        }
      }

      for (int i = 0; i < static_cast<int>(select_specs.size()); ++i) {
        auto action = LegalAction::from_spec(select_specs[i]);
        action.response_ = core_select_unselect_response_index(
            select_specs.size(), unselect_specs.size(), false, i);
        legal_actions_.push_back(std::move(action));
      }
      for (int i = 0; i < static_cast<int>(unselect_specs.size()); ++i) {
        auto action = LegalAction::from_spec(unselect_specs[i]);
        action.unselect_ = true;
        action.response_ = core_select_unselect_response_index(
            select_specs.size(), unselect_specs.size(), true, i);
        legal_actions_.push_back(std::move(action));
      }
      if (finishable) {
        legal_actions_.push_back(LegalAction::finish());
      } else if (cancelable) {
        legal_actions_.push_back(LegalAction::cancel());
      }

      to_play_ = player;
      callback_ = [this](int idx) {
        const auto &action = legal_actions_[idx];
        if (action.finish_ || action.act_ == ActionAct::Cancel) {
          YGO_SetResponsei(pduel_, -1);
        } else {
          resp_buf_[0] = 1;
          resp_buf_[1] = static_cast<uint8_t>(action.response_);
          YGO_SetResponseb(pduel_, resp_buf_);
        }
      };

    } else if (msg_ == MSG_SELECT_CARD) {
      auto player = read_u8();
      bool cancelable = read_u8();
      auto min = read_u8();
      auto max = read_u8();
      auto size = read_u8();
      selection_cancelable_ = cancelable;
      selection_finishable_ = min == 0;

      std::vector<std::string> specs;
      specs.reserve(size);
      if (verbose_) {
        std::vector<Card> cards;
        for (int i = 0; i < size; ++i) {
          auto code = read_u32();
          auto loc = read_u32();
          Card card = c_get_card(code);
          card.set_location(loc);
          cards.push_back(card);
        }
        auto& pl = players_[player];
        pl->notify("Select " + std::to_string(min) + " to " +
                   std::to_string(max) + " cards separated by spaces:");
        for (const auto &card : cards) {
          auto spec = card.get_spec(player);
          specs.push_back(spec);
          int i = specs.size();
          if (card.controler_ != player && card.position_ & POS_FACEDOWN) {
            pl->notify(
              fmt::format("{}: {} card ({})", i, card.get_position(), spec));
          } else {
            pl->notify(
              fmt::format("{}: {} ({})", i, card.name_, spec));
          }
        }
      } else {
        for (int i = 0; i < size; ++i) {
          dp_ += 4;
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto pos = read_u8();
          auto spec = ls_to_spec(loc, seq, pos, controller != player);
          specs.push_back(spec);
        }
      }

      if (discard_hand_) {
        discard_hand_ = false;
        if (current_phase_ == PHASE_END) {
          // random discard
          std::vector<int> comb(size);
          std::iota(comb.begin(), comb.end(), 0);
          std::shuffle(comb.begin(), comb.end(), gen_);
          resp_buf_[0] = min;
          for (int i = 0; i < min; ++i) {
            resp_buf_[i + 1] = comb[i];
          }
          YGO_SetResponseb(pduel_, resp_buf_);
          return;
        }
      }

      // TODO(1): use this when added to history actions
      // if ((min == max) && (max == specs.size())) {
      //   resp_buf_[0] = specs.size();
      //   for (int i = 0; i < specs.size(); ++i) {
      //     resp_buf_[i + 1] = i;
      //   }
      //   YGO_SetResponseb(pduel_, resp_buf_);
      //   return;
      // }

      init_multi_select(min, max, 0, specs);

      to_play_ = player;
      callback_ = [this](int idx) {
        _callback_multi_select(idx, ms_max_ == 1);
      };
    } else if (msg_ == MSG_SELECT_TRIBUTE) {
      auto player = read_u8();
      bool cancelable = read_u8();
      auto min = read_u8();
      auto max = read_u8();
      auto size = read_u8();
      selection_cancelable_ = cancelable;
      selection_finishable_ = min == 0;

      std::vector<int> release_params;
      release_params.reserve(size);
      std::vector<std::string> specs;
      specs.reserve(size);
      if (verbose_) {
        std::vector<Card> cards;
        for (int i = 0; i < size; ++i) {
          auto code = read_u32();
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto release_param = read_u8();
          Card card = get_card(controller, loc, seq);
          cards.push_back(card);
          release_params.push_back(release_param);
        }
        auto& pl = players_[player];
        pl->notify("Select " + std::to_string(min) + " to " +
                   std::to_string(max) +
                   " cards to tribute separated by spaces:");
        for (const auto &card : cards) {
          auto spec = card.get_spec(player);
          specs.push_back(spec);
          pl->notify(
            fmt::format("{}: {} ({})", specs.size(), card.name_, spec));
        }
      } else {
        for (int i = 0; i < size; ++i) {
          dp_ += 4;
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto release_param = read_u8();

          auto spec = ls_to_spec(loc, seq, 0, controller != player);
          specs.push_back(spec);

          release_params.push_back(release_param);
        }
      }

      bool has_weight =
          std::any_of(release_params.begin(), release_params.end(),
                      [](int i) { return i != 1; });

      if (has_weight || min != max || min == 0) {
        ms_weighted_kind_ = CoreWeightedKind::Tribute;
        ms_weighted_must_.clear();
        ms_weighted_optional_.assign(release_params.begin(),
                                     release_params.end());
        ms_weighted_target_ = min;
        init_multi_select(min, max, 0, specs, 3);
        to_play_ = player;
        callback_ = [this](int idx) { callback_weighted_selection(idx); };
        return;
      }

      // TODO(1): use this when added to history actions
      // if (max == specs.size()) {
      //   // tribute all
      //   resp_buf_[0] = specs.size();
      //   for (int i = 0; i < specs.size(); ++i) {
      //     resp_buf_[i + 1] = i;
      //   }
      //   YGO_SetResponseb(pduel_, resp_buf_);
      //   return;
      // }

      init_multi_select(min, max, 0, specs);

      to_play_ = player;
      callback_ = [this](int idx) {
        _callback_multi_select(idx, ms_max_ == 1);
      };
    } else if (msg_ == MSG_SELECT_SUM) {
      auto mode = read_u8();
      auto player = read_u8();
      int32_t target = read_u32();
      int min_count = read_u8();
      int max_count = read_u8();
      auto must_select_size = read_u8();
      std::vector<uint32_t> must_params;
      must_params.reserve(must_select_size);
      std::vector<std::string> must_specs;
      must_specs.reserve(must_select_size);
      for (int i = 0; i < must_select_size; ++i) {
        auto code = read_u32();
        auto controller = read_u8();
        auto loc = read_u8();
        auto seq = read_u8();
        auto param = read_u32();
        must_params.push_back(param);
        must_specs.push_back(ls_to_spec(loc, seq, 0, controller != player));
        if (verbose_) {
          players_[player]->notify(
              c_get_card(code).name_ + " (" + must_specs.back() +
              ") must be selected, automatically selected.");
        }
      }

      uint8_t select_size = read_u8();
      std::vector<uint32_t> select_params;
      std::vector<std::string> select_specs;
      select_params.reserve(select_size);
      select_specs.reserve(select_size);
      for (int i = 0; i < select_size; ++i) {
        auto code = read_u32();
        auto controller = read_u8();
        auto loc = read_u8();
        auto seq = read_u8();
        auto param = read_u32();
        select_params.push_back(param);
        select_specs.push_back(ls_to_spec(loc, seq, 0, controller != player));
        if (verbose_) {
          players_[player]->notify(fmt::format(
              "{}: {} ({}) values {}/{}", select_specs.size(),
              c_get_card(code).name_, select_specs.back(), param & 0xffff,
              param >> 16));
        }
      }

      if (mode == 0) {
        ms_weighted_kind_ = CoreWeightedKind::ExactSum;
      } else if (mode == 1) {
        ms_weighted_kind_ = CoreWeightedKind::SumLimit;
        min_count = 0;
        max_count = select_size;
      } else {
        throw std::runtime_error(
            "Invalid MSG_SELECT_SUM mode " + std::to_string(mode));
      }
      ms_weighted_must_ = std::move(must_params);
      ms_weighted_optional_ = std::move(select_params);
      ms_weighted_target_ = target;
      init_multi_select(min_count, max_count, must_select_size, select_specs, 3);

      to_play_ = player;
      callback_ = [this](int idx) { callback_weighted_selection(idx); };

    } else if (msg_ == MSG_SELECT_CHAIN) {
      auto player = read_u8();
      auto size = read_u8();
      auto spe_count = read_u8();
      bool forced = read_u8();
      selection_forced_ = forced;
      selection_cancelable_ = !forced;
      dp_ += 8;
      // auto hint_timing = read_u32();
      // auto other_timing = read_u32();

      std::vector<CardCode> codes;
      std::vector<uint32_t> descs;
      std::vector<std::string> specs;
      for (int i = 0; i < size; ++i) {
        auto flag = read_u8();
        CardCode code = read_u32();
        codes.push_back(code);
        PlayerId c = read_u8();
        uint8_t loc = read_u8();
        uint8_t seq = read_u8();
        uint8_t pos = read_u8();
        specs.push_back(ls_to_spec(loc, seq, pos, c != player));
        uint32_t desc = read_u32();
        descs.push_back(desc);
      }

      if ((size == 0) && (spe_count == 0)) {
        // non-GUI don't need this
        // if (verbose_) {
        //   fmt::println("keep processing");
        // }
        YGO_SetResponsei(pduel_, -1);
        return;
      }

      auto& pl = players_[player];
      auto& op = players_[1 - player];
      chaining_player_ = player;
      if (!op->seen_waiting_) {
        if (verbose_) {
          op->notify("Waiting for opponent.");
        }
        op->seen_waiting_ = true;
      }

      if (verbose_) {
        pl->notify("Select chain:");
      }

      for (int i = 0; i < size; i++) {
        CardCode code = codes[i];
        uint32_t desc = descs[i];
        auto spec = specs[i];
        auto [code_d, eff_idx] = unpack_desc(code, desc);
        if (desc == 0) {
          code_d = code;
        }
        auto la = LegalAction::activate_spec(eff_idx, spec);
        if (code_d != 0) {
          la.cid_ = c_get_card_id(code_d);
        }
        legal_actions_.push_back(la);
        if (verbose_) {
          auto c = c_get_card(code);
          std::string s = fmt::format(
            "{}: {}({}) ({})",
            i + 1, c.name_, spec, c.get_effect_description(code_d, eff_idx));
          pl->notify(s);
        }
      }

      if (!forced) {
        legal_actions_.push_back(LegalAction::cancel());
        if (verbose_) {
          pl->notify(fmt::format("{}: cancel", size + 1));
        }
      }
      to_play_ = player;
      callback_ = [this, forced](int idx) {
        const auto &action = legal_actions_[idx];
        if (action.act_ == ActionAct::Cancel) {
          if (forced) {
            fmt::print("cancel not allowed in forced chain\n");
            YGO_SetResponsei(pduel_, 0);
            return;
          }
          YGO_SetResponsei(pduel_, -1);
          return;
        }
        YGO_SetResponsei(pduel_, idx);
      };
    } else if (msg_ == MSG_SELECT_YESNO) {
      selection_cancelable_ = true;
      auto player = read_u8();
      auto desc = read_u32();
      auto [code, eff_idx] = unpack_desc(0, desc);
      if (desc == 0) {
        show_buffer();
        auto s = fmt::format("Unknown desc {} in select_yesno", desc);
        throw std::runtime_error(s);
      }
      auto la = LegalAction::activate_spec(eff_idx, "");
      if (code != 0) {
        la.cid_ = c_get_card_id(code);
      }
      legal_actions_.push_back(la);
      if (verbose_) {
        auto& pl = players_[player];
        std::string s;
        if (code == 0) {
          s = get_system_string(eff_idx);
        } else {
          Card c = c_get_card(code);
          int cmd_idx = legal_actions_.size();
          eff_idx -= CARD_EFFECT_OFFSET;
          if (eff_idx >= c.strings_.size()) {
            throw std::runtime_error(
              fmt::format("Unknown effect {} of {}", eff_idx, c.name_));
          }
          auto str = c.strings_[eff_idx];
          if (str.empty()) {
            str = "effect " + std::to_string(eff_idx);
          }
          s = fmt::format("{} ({})", c.name_, str);
        }
        pl->notify("1: " + s);
        pl->notify("2: No");
      }
      // TODO: maybe add card id to cancel
      legal_actions_.push_back(LegalAction::cancel());
      to_play_ = player;
      callback_ = [this](int idx) {
        if (idx == 0) {
          YGO_SetResponsei(pduel_, 1);
        } else if (idx == 1) {
          YGO_SetResponsei(pduel_, 0);
        }
      };
    } else if (msg_ == MSG_SELECT_EFFECTYN) {
      selection_cancelable_ = true;
      auto player = read_u8();

      CardCode code = read_u32();
      auto ct = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      auto pos = read_u8();
      auto desc = read_u32();
      std::string spec = ls_to_spec(loc, seq, pos, ct != player);
      auto [code_d, eff_idx] = unpack_desc(code, desc);
      if (desc == 0) {
        code_d = code;
      }
      auto la = LegalAction::activate_spec(eff_idx, spec);
      if (code_d != 0) {
        la.cid_ = c_get_card_id(code_d);
      }
      legal_actions_.push_back(la);

      if (verbose_) {
        Card c = c_get_card(code);
        auto& pl = players_[player];
        auto name = c.name_;
        std::string s;
        if (code_d == 0) {
          s = get_system_string(desc);
          std::string fmt_str = "[%ls]";
          auto pos = find_substrs(s, fmt_str);
          if (pos.size() == 0) {
            // nothing to replace
          } else if (pos.size() == 1) {
            auto p = pos[0];
            s = s.substr(0, p) + name + s.substr(p + fmt_str.size());
          } else if (pos.size() == 2) {
            auto p1 = pos[0];
            auto p2 = pos[1];
            s = s.substr(0, p1) + spec +
                s.substr(p1 + fmt_str.size(), p2 - p1 - fmt_str.size()) + name +
                s.substr(p2 + fmt_str.size());
          } else {
            throw std::runtime_error("Unknown effectyn desc " +
                                     std::to_string(desc) + " of " + name);
          }
        } else {
          s = fmt::format(
            "{}({}) ({})", c.name_, spec, c.get_effect_description(code_d, eff_idx));
        }
        pl->notify("1: " + s);
        pl->notify("2: No");
      }

      // TODO: maybe add card info to cancel
      legal_actions_.push_back(LegalAction::cancel());
      to_play_ = player;
      callback_ = [this](int idx) {
        if (idx == 0) {
          YGO_SetResponsei(pduel_, 1);
        } else if (idx == 1) {
          YGO_SetResponsei(pduel_, 0);
        }
      };
    } else if (msg_ == MSG_SELECT_OPTION) {
      auto player = read_u8();
      auto size = read_u8();
      if (verbose_) {
        players_[player]->notify("Select an option:");
      }
      for (int i = 0; i < size; ++i) {
        auto desc = read_u32();
        auto [code, eff_idx] = unpack_desc(0, desc);
        if (desc == 0) {
          show_buffer();
          auto s = fmt::format("Unknown desc {} in select_option", desc);
          throw std::runtime_error(s);
        }
        auto la = LegalAction::activate_spec(eff_idx, "");
        if (code != 0) {
          la.cid_ = c_get_card_id(code);
        }
        legal_actions_.push_back(la);
        if (verbose_) {
          std::string s;
          if (code == 0) {
            s = get_system_string(eff_idx);
          } else {
            Card c = c_get_card(code);
            int cmd_idx = legal_actions_.size();
            eff_idx -= CARD_EFFECT_OFFSET;
            if (eff_idx >= c.strings_.size()) {
              throw std::runtime_error(
                fmt::format("Unknown effect {} of {}", eff_idx, c.name_));
            }
            auto str = c.strings_[eff_idx];
            if (str.empty()) {
              str = "effect " + std::to_string(eff_idx);
            }
            s = fmt::format("{} ({})", c.name_, str);
          }
          players_[player]->notify(std::to_string(i + 1) + ": " + s);
        }
      }

      to_play_ = player;
      callback_ = [this](int idx) {
        YGO_SetResponsei(pduel_, idx);
      };
    } else if (msg_ == MSG_SELECT_IDLECMD) {
      int32_t player = read_u8();
      auto summonable_ = read_cardlist_spec(player);
      auto spsummon_ = read_cardlist_spec(player);
      auto repos_ = read_cardlist_spec(player);
      auto idle_mset_ = read_cardlist_spec(player);
      auto idle_set_ = read_cardlist_spec(player);
      auto idle_activate_ = read_cardlist_spec(player, true);
      bool to_bp_ = read_u8();
      bool to_ep_ = read_u8();
      bool can_shuffle_ = read_u8();

      int offset = 0;

      auto& pl = players_[player];
      if (verbose_) {
        pl->notify("Select a card and action to perform.");
      }
      for (const auto &[code, spec, data] : summonable_) {
        legal_actions_.push_back(LegalAction::act_spec(ActionAct::Summon, spec));
        if (verbose_) {
          const auto &name = c_get_card(code).name_;
          int cmd_idx = legal_actions_.size();
          pl->notify(fmt::format(
            "{}: Summon {} in face-up attack position", cmd_idx, name));
        }
      }
      offset += summonable_.size();
      int spsummon_offset = offset;
      for (const auto &[code, spec, data] : spsummon_) {
        legal_actions_.push_back(LegalAction::act_spec(ActionAct::SpSummon, spec));
        if (verbose_) {
          const auto &name = c_get_card(code).name_;
          int cmd_idx = legal_actions_.size();
          pl->notify(fmt::format(
            "{}: Special summon {}", cmd_idx, name));
        }
      }
      offset += spsummon_.size();
      int repos_offset = offset;
      for (const auto &[code, spec, data] : repos_) {
        legal_actions_.push_back(LegalAction::act_spec(ActionAct::Repo, spec));
        if (verbose_) {
          const auto &name = c_get_card(code).name_;
          int cmd_idx = legal_actions_.size();
          pl->notify(fmt::format(
            "{}: Change position of {}", cmd_idx, name));
        }
      }
      offset += repos_.size();
      int mset_offset = offset;
      for (const auto &[code, spec, data] : idle_mset_) {
        legal_actions_.push_back(LegalAction::act_spec(ActionAct::MSet, spec));
        if (verbose_) {
          const auto &name = c_get_card(code).name_;
          int cmd_idx = legal_actions_.size();
          pl->notify(fmt::format(
            "{}: Summon {} in face-down defense position", cmd_idx, name));
        }
      }
      offset += idle_mset_.size();
      int set_offset = offset;
      for (const auto &[code, spec, data] : idle_set_) {
        legal_actions_.push_back(LegalAction::act_spec(ActionAct::Set, spec));
        if (verbose_) {
          const auto &name = c_get_card(code).name_;
          int cmd_idx = legal_actions_.size();
          pl->notify(fmt::format(
            "{}: Set {}", cmd_idx, name));
        }
      }
      offset += idle_set_.size();
      int activate_offset = offset;
      for (const auto &[code_t, spec, desc] : idle_activate_) {
        CardCode code = code_t;
        if(code & 0x80000000) {
          code &= 0x7fffffff;
        }
        auto [code_d, eff_idx] = unpack_desc(code, desc);
        if (desc == 0) {
          code_d = code;
        }
        auto la = LegalAction::activate_spec(eff_idx, spec);
        if (code_d != 0) {
          la.cid_ = c_get_card_id(code_d);
        }
        legal_actions_.push_back(la);
        if (verbose_) {
          auto c = c_get_card(code);
          int cmd_idx = legal_actions_.size();
          std::string s = fmt::format(
            "{}: Activate {}({}) ({})",
            cmd_idx, c.name_, spec, c.get_effect_description(code_d, eff_idx));
          pl->notify(s);
        }
      }

      if (to_bp_) {
        legal_actions_.push_back(LegalAction::phase(ActionPhase::Battle));
        if (verbose_) {
          int cmd_idx = legal_actions_.size();
          pl->notify(fmt::format("{}: Enter the battle phase.", cmd_idx));
        }
      }
      if (to_ep_) {
        legal_actions_.push_back(LegalAction::phase(ActionPhase::End));
        if (verbose_) {
          int cmd_idx = legal_actions_.size();
          pl->notify(fmt::format("{}: End phase.", cmd_idx));
        }
      }
      if (can_shuffle_) {
        // The 40M model has only four frozen phase rows.  The idle-menu
        // shuffle command is an action, not a new phase embedding row.
        legal_actions_.push_back(LegalAction::finish());
      }

      to_play_ = player;
      callback_ = [this, spsummon_offset, repos_offset, mset_offset, set_offset,
                   activate_offset](int idx) {
        const auto &action = legal_actions_[idx];
        if (action.phase_ == ActionPhase::Battle) {
          YGO_SetResponsei(pduel_, 6);
        } else if (action.phase_ == ActionPhase::End) {
          YGO_SetResponsei(pduel_, 7);
        } else if (action.finish_) {
          YGO_SetResponsei(pduel_, 8);
        } else {
          auto act = action.act_;
          if (act == ActionAct::Summon) {
            uint32_t idx_ = idx;
            YGO_SetResponsei(pduel_, idx_ << 16);
          } else if (act == ActionAct::SpSummon) {
            uint32_t idx_ = idx - spsummon_offset;
            YGO_SetResponsei(pduel_, (idx_ << 16) + 1);
          } else if (act == ActionAct::Repo) {
            uint32_t idx_ = idx - repos_offset;
            YGO_SetResponsei(pduel_, (idx_ << 16) + 2);
          } else if (act == ActionAct::MSet) {
            uint32_t idx_ = idx - mset_offset;
            YGO_SetResponsei(pduel_, (idx_ << 16) + 3);
          } else if (act == ActionAct::Set) {
            uint32_t idx_ = idx - set_offset;
            YGO_SetResponsei(pduel_, (idx_ << 16) + 4);
          } else if (act == ActionAct::Activate) {
            uint32_t idx_ = idx - activate_offset;
            YGO_SetResponsei(pduel_, (idx_ << 16) + 5);
          }
        }
      };
    } else if (msg_ == MSG_SELECT_PLACE || msg_ == MSG_SELECT_DISFIELD) {
      // TODO(1): add card informaton to select place
      auto player = read_u8();
      const auto protocol_count = read_u8();
      const int count = std::max(1, static_cast<int>(protocol_count));
      auto flag = read_u32();
      auto places = flag_to_usable_places(flag);
      if (protocol_count > places.size()) {
        throw std::runtime_error(fmt::format(
            "MSG_SELECT_PLACE requests {} distinct places from {} candidates",
            protocol_count, places.size()));
      }
      if (verbose_) {
        auto place_s = msg_ == MSG_SELECT_PLACE ? "place" : "disfield";
        auto s = fmt::format("Select {} {} for card, from:", count, place_s);
        players_[player]->notify(s);
      }
      for (int i = 0; i < places.size(); ++i) {
        legal_actions_.push_back(LegalAction::place(places[i]));
        if (verbose_) {
          auto s = fmt::format("{}: {}", i + 1, action_place_to_string(places[i]));
          players_[player]->notify(s);
        }
      }
      selection_finishable_ = protocol_count == 0;
      if (selection_finishable_) legal_actions_.push_back(LegalAction::finish());
      ms_mode_ = 2;
      ms_idx_ = 0;
      ms_min_ = protocol_count;
      ms_max_ = count;
      ms_must_ = 0;
      ms_place_player_ = player;
      n_places_ = count;
      ms_places_ = places;
      ms_selected_places_.clear();
      to_play_ = player;
      callback_ = [this](int idx) { callback_place_select(idx); };
    } else if (msg_ == MSG_SELECT_COUNTER) {
      auto player = read_u8();
      auto counter_type = read_u16();
      int counter_count = read_u16();
      int count = read_u8();
      n_counters_ = count;
      if (count <= 0) {
        throw std::runtime_error("Select counter requires at least one card");
      }
      auto& pl = players_[player];
      if (verbose_) {
        pl->notify(fmt::format("Type new {} for {} card(s), separated by spaces.", "UNKNOWN_COUNTER", count));
      }
      std::vector<int> counters;
      counters.reserve(count);
      for (int i = 0; i < count; ++i) {
        auto code = read_u32();
        auto controller = read_u8();
        auto loc = read_u8();
        auto seq = read_u8();
        auto counter = read_u16();
        counters.push_back(counter & 0xffff);

        if (verbose_) {
          pl->notify(c_get_card(code).name_ + ": " + std::to_string(counter));
        }
        // auto spec = ls_to_spec(loc, seq, 0, controller != player);
        // options_.push_back(spec);
      }
      // Split each feasible uint16 allocation range into binary decisions.
      // Every allocation stays reachable with two actions per model step.
      ms_mode_ = 5;
      ms_idx_ = 0;
      ms_counter_capacities_ = std::move(counters);
      ms_counter_allocations_.clear();
      ms_counter_remaining_ = counter_count;
      ms_counter_card_ = 0;
      ms_counter_low_ = 0;
      ms_counter_high_ = 0xffff;
      selection_finishable_ = false;
      prepare_counter_selection();
      to_play_ = player;
      callback_ = [this](int idx) { callback_counter_selection(idx); };
    } else if (msg_ == MSG_ANNOUNCE_NUMBER) {
      auto player = read_u8();
      int count = read_u8();
      std::vector<uint32_t> numbers;
      for (int i = 0; i < count; ++i) {
        uint32_t number = read_u32();
        if (number == 0) {
          throw std::runtime_error(
              "Invalid non-positive announce number " + std::to_string(number));
        }
        numbers.push_back(number);
        legal_actions_.push_back(LegalAction::number(number));
      }
      if (verbose_) {
        auto& pl = players_[player];
        std::string str = "Select a number, one of:";
        pl->notify(str);
        for (int i = 0; i < count; ++i) {
          pl->notify(fmt::format("{}: {}", i + 1, numbers[i]));
        }
      }
      to_play_ = player;
      callback_ = [this](int idx) {
        YGO_SetResponsei(pduel_, idx);
      };
    } else if (msg_ == MSG_ANNOUNCE_RACE) {
      auto player = read_u8();
      int count = read_u8();
      uint32_t flag = read_u32();
      std::vector<uint32_t> races;
      // The pinned core defines exactly RACES_COUNT race bits. Higher bits in
      // the uint32 payload are ignored by its validator and are not actions.
      for (int bit = 0; bit < RACES_COUNT; ++bit) {
        uint32_t race = uint32_t{1} << bit;
        if (flag & race) races.push_back(race);
      }
      if (count < 0 || count > static_cast<int>(races.size())) {
        throw std::runtime_error(fmt::format(
            "Invalid announce race count {} for {} candidates", count, races.size()));
      }
      core_validate_action_capacity(
          core_bounded_combination_count(
              static_cast<int>(races.size()), count, max_options()),
          max_options());
      auto combs = combinations(static_cast<int>(races.size()), count);
      for (const auto &comb : combs) {
        uint32_t response = 0;
        for (int index : comb) response |= races[index];
        legal_actions_.push_back(LegalAction::race(response));
      }
      if (verbose_) {
        auto &pl = players_[player];
        pl->notify(fmt::format("Select {} race(s):", count));
        for (int i = 0; i < static_cast<int>(legal_actions_.size()); ++i) {
          pl->notify(fmt::format("{}: race mask 0x{:08x}", i + 1,
                                legal_actions_[i].race_));
        }
      }
      to_play_ = player;
      callback_ = [this](int idx) {
        YGO_SetResponsei(pduel_, legal_actions_[idx].race_);
      };
    } else if (msg_ == MSG_ANNOUNCE_ATTRIB) {
      auto player = read_u8();
      int count = read_u8();
      auto flag = read_u32();

      int n_attrs = 7;

      std::vector<uint8_t> attrs;
      for (int i = 0; i < n_attrs; i++) {
        if (flag & (1 << i)) {
          attrs.push_back(i + 1);
        }
      }
      if (count < 0 || count > static_cast<int>(attrs.size())) {
        throw std::runtime_error(fmt::format(
          "Invalid announce attribute count {} for {} candidates", count, attrs.size()));
      }
      core_validate_action_capacity(
          core_bounded_combination_count(
              static_cast<int>(attrs.size()), count, max_options()),
          max_options());

      if (verbose_) {
        auto& pl = players_[player];
        pl->notify("Select " + std::to_string(count) +
                   " attributes separated by spaces:");
        for (int i = 0; i < attrs.size(); i++) {
          pl->notify(fmt::format("{}: {}", i + 1, attribute_to_string(1 << (attrs[i] - 1))));
        }
      }

      auto combs = combinations(static_cast<int>(attrs.size()), count);
      for (const auto &comb : combs) {
        int response = 0;
        for (int index : comb) response |= 1 << (attrs[index] - 1);
        legal_actions_.push_back(LegalAction::attribute(response));
      }

      to_play_ = player;
      callback_ = [this](int idx) {
        const auto &action = legal_actions_[idx];
        YGO_SetResponsei(pduel_, action.attribute_);
      };
    } else if (msg_ == MSG_ANNOUNCE_CARD) {
      auto player = read_u8();
      int count = read_u8();

      std::vector<uint32_t> opcodes;
      opcodes.reserve(count);
      for (int i = 0; i < count; i++) {
        opcodes.push_back(read_u32());
      }

      std::vector<uint32_t> codes;
      std::unordered_set<uint32_t> seen;
      // Put explicitly named cards first so a mixed broad/named expression
      // cannot lose its named alternatives when the environment applies the
      // fixed max-options cap.
      for (int i = 1; i < static_cast<int>(opcodes.size()); ++i) {
        if (opcodes[i] != OPCODE_ISCODE) continue;
        uint32_t candidate = opcodes[i - 1];
        auto it = cards_data_.find(candidate);
        if (it != cards_data_.end() && is_declarable(it->second, opcodes) &&
            seen.insert(candidate).second) {
          codes.push_back(candidate);
        }
      }
      std::vector<uint32_t> remaining;
      remaining.reserve(cards_data_.size());
      for (const auto &[candidate, data] : cards_data_) {
        if (candidate != 0 && !seen.count(candidate) &&
            is_declarable(data, opcodes)) {
          remaining.push_back(candidate);
        }
      }
      std::sort(remaining.begin(), remaining.end());
      codes.insert(codes.end(), remaining.begin(), remaining.end());
      if (codes.empty()) {
        throw std::runtime_error("announce card filter has no declarable candidates");
      }

      if (verbose_) {
        auto& pl = players_[player];
        pl->notify(fmt::format("Select 1 declarable card from {} candidates.",
                               codes.size()));
      }
      ms_announce_codes_ = std::move(codes);
      ms_announce_lo_ = 0;
      ms_announce_hi_ = ms_announce_codes_.size();
      ms_mode_ = 6;
      ms_idx_ = 0;
      to_play_ = player;
      prepare_announce_card_selection();
    } else if (msg_ == MSG_SELECT_POSITION) {
      auto player = read_u8();
      auto code = read_u32();
      auto valid_pos = read_u8();
      CardId cid = c_get_card_id(code);

      if (verbose_) {
        auto& pl = players_[player];
        auto card = c_get_card(code);
        pl->notify("Select position for " + card.name_ + ":");
      }

      for (auto pos : {POS_FACEUP_ATTACK, POS_FACEDOWN_ATTACK,
                       POS_FACEUP_DEFENSE, POS_FACEDOWN_DEFENSE}) {
        if (valid_pos & pos) {
          LegalAction la;
          la.cid_ = cid;
          la.position_ = pos;
          legal_actions_.push_back(la);
          int cmd_idx = legal_actions_.size();
          if (verbose_) {
            auto& pl = players_[player];
            pl->notify(fmt::format("{}: {}", cmd_idx, position_to_string(pos)));
          }
        }
      }

      to_play_ = player;
      callback_ = [this](int idx) {
        uint8_t pos = legal_actions_[idx].position_;
        YGO_SetResponsei(pduel_, pos);
      };
    } else {
      show_deck(0);
      show_deck(1);
      show_buffer();
      throw std::runtime_error(
        fmt::format("Unknown message {}, length {}, dp {}",
        msg_to_string(msg_), dl_, dp_));
    }
  }

  void _damage(uint8_t player, uint32_t amount) {
    lp_[player] -= amount;
    if (verbose_) {
      auto& lp = players_[player];
      lp->notify(fmt::format("Your lp decreased by {}, now {}", amount, lp_[player]));
      players_[1 - player]->notify(fmt::format("{}'s lp decreased by {}, now {}",
                                   lp->nickname_, amount, lp_[player]));
    }
  }

  void _recover(uint8_t player, uint32_t amount) {
    lp_[player] += amount;
    if (verbose_) {
      auto& lp = players_[player];
      lp->notify(fmt::format("Your lp increased by {}, now {}", amount, lp_[player]));
      players_[1 - player]->notify(fmt::format("{}'s lp increased by {}, now {}",
                                   lp->nickname_, amount, lp_[player]));
    }
  }

  void _duel_end(uint8_t player, uint8_t reason) {
    winner_ = player;
    win_reason_ = reason;
    termination_reason_ = kTerminationNatural;
    if (play_mode_ == kWindBot && windbot_fd_ >= 0) {
      try { windbot_send(0x16); } catch (...) {}
#ifndef _WIN32
      ::close(windbot_fd_);
      windbot_fd_ = -1;
#endif
      windbot_lobby_ready_ = false;
    }
    YGO_EndDuel(pduel_);

    duel_started_ = false;
  }
};

// Test-only bridge: feed a frame emitted by the linked core into the actual
// adapter parser and response callback, while retaining the same duel for the
// core's step-1 response validator.
class ProtocolAdapterProbe : public YGOProEnvImpl {
public:
  ProtocolAdapterProbe() : YGOProEnvImpl(YGOProEnvSpec{}, 0) {
    for (int player = 0; player < 2; ++player) {
      if (!players_[player]) {
        players_[player] = std::make_unique<GreedyAI>(
            "protocol-probe", 8000, player);
      }
    }
  }

  void attach(duel &fixture, const std::vector<uint8_t> &frame) {
    if (frame.empty() || frame.size() > sizeof(data_)) {
      throw std::runtime_error("Invalid adapter probe frame size");
    }
    pduel_ = reinterpret_cast<intptr_t>(&fixture);
    std::copy(frame.begin(), frame.end(), data_);
    dp_ = 0;
    dl_ = static_cast<int>(frame.size());
    ms_idx_ = -1;
    handle_message();
  }

  size_t choices() const { return legal_actions_.size(); }

  std::tuple<int, int, bool, bool> policy_cancel_filter_fixture(int msg) {
    msg_ = msg;
    play_mode_ = kSelfPlay;
    selection_cancelable_ = true;
    policy_back_cancel_suppressed_ = false;
    legal_actions_ = {
        LegalAction::from_spec("h1"), LegalAction::cancel()};
    const int before = static_cast<int>(legal_actions_.size());
    filter_policy_navigational_back();
    const bool has_cancel = std::any_of(
        legal_actions_.begin(), legal_actions_.end(),
        [](const LegalAction &action) {
          return action.act_ == ActionAct::Cancel;
        });
    return {before, static_cast<int>(legal_actions_.size()), has_cancel,
            policy_back_cancel_suppressed_};
  }

  void assert_frozen_command_domains() const {
    for (const auto &action : legal_actions_) {
      if (static_cast<int>(action.phase_) >= 4 ||
          static_cast<int>(action.act_) >= 10) {
        throw std::runtime_error("Protocol command exceeds frozen action domain");
      }
    }
  }

  void choose(int index) {
    if (index < 0 || index >= static_cast<int>(legal_actions_.size())) {
      throw std::runtime_error("Invalid adapter probe action index");
    }
    callback_(index);
    if (ms_idx_ != -1) handle_multi_select();
  }

  bool complete() const { return ms_idx_ == -1; }

  std::vector<uint8_t> encode_history_action(const LegalAction &action) {
    TArray<uint8_t> features(
        Array(ShapeSpec(sizeof(uint8_t), {1, 14})));
    features.Zero();
    _set_obs_action(features, 0, action);
    auto *begin = static_cast<uint8_t *>(features.Data());
    return std::vector<uint8_t>(begin, begin + 14);
  }

  std::vector<int> parse_notifications(
      duel &fixture, const std::vector<uint8_t> &frame) {
    if (frame.empty() || frame.size() > sizeof(data_)) {
      throw std::runtime_error("Invalid adapter notification frame size");
    }
    pduel_ = reinterpret_cast<intptr_t>(&fixture);
    std::copy(frame.begin(), frame.end(), data_);
    dp_ = 0;
    dl_ = static_cast<int>(frame.size());
    ms_idx_ = -1;
    std::vector<int> messages;
    while (dp_ < dl_) {
      const int previous = dp_;
      handle_message();
      messages.push_back(msg_);
      if (!legal_actions_.empty() || ms_idx_ != -1) {
        throw std::runtime_error(
            "Interactive message in notification adapter fixture");
      }
      if (dp_ <= previous) {
        throw std::runtime_error(
            "Notification parser made no forward progress");
      }
    }
    if (dp_ != dl_) {
      throw std::runtime_error("Notification parser crossed frame boundary");
    }
    return messages;
  }
};

inline std::vector<uint8_t> core_cancel_history_encoding_fixture(int msg) {
  ProtocolAdapterProbe probe;
  LegalAction action = LegalAction::cancel();
  action.msg_ = msg;
  return probe.encode_history_action(action);
}

inline std::tuple<int, int, bool, bool>
core_policy_cancel_filter_fixture(int msg) {
  ProtocolAdapterProbe probe;
  return probe.policy_cancel_filter_fixture(msg);
}

inline std::vector<int> core_notification_adapter_fixture(
    const std::vector<uint8_t> &frame) {
  duel fixture;
  ProtocolAdapterProbe probe;
  return probe.parse_notifications(fixture, frame);
}

inline std::tuple<std::vector<uint8_t>, std::vector<int>, bool>
core_sort_adapter_fixture(int count, const std::vector<int> &choices) {
  if (count < 1 || count > 32) {
    throw std::runtime_error("Invalid adapter sort fixture size");
  }
  duel fixture;
  for (int index = 0; index < count; ++index) {
    card *pcard = fixture.new_card(0);
    pcard->data.code = 1000 + index;
    pcard->current.controler = 0;
    pcard->current.location = LOCATION_HAND;
    pcard->current.sequence = index;
    fixture.game_field->core.select_cards.push_back(pcard);
  }
  fixture.game_field->sort_card(0, 0);
  std::vector<uint8_t> frame(fixture.message_buffer.begin(),
                             fixture.message_buffer.end());
  fixture.clear_buffer();
  ProtocolAdapterProbe probe;
  probe.attach(fixture, frame);
  std::vector<int> branch_sizes;
  for (int choice : choices) {
    branch_sizes.push_back(static_cast<int>(probe.choices()));
    probe.choose(choice);
  }
  if (!probe.complete()) {
    throw std::runtime_error("Incomplete adapter sort fixture sequence");
  }
  const bool accepted = fixture.game_field->sort_card(1, 0);
  return {frame, branch_sizes, accepted};
}

inline std::tuple<std::vector<uint8_t>, std::vector<int>,
                  std::vector<uint16_t>, bool>
core_counter_adapter_fixture(const std::vector<uint16_t> &capacities,
                             uint16_t requested,
                             const std::vector<int> &choices) {
  if (capacities.empty() || capacities.size() > 7) {
    throw std::runtime_error("Invalid adapter counter fixture size");
  }
  constexpr uint16_t counter_type = 1;
  duel fixture;
  for (int index = 0; index < static_cast<int>(capacities.size()); ++index) {
    card *pcard = fixture.new_card(0);
    pcard->data.code = 1000 + index;
    pcard->current.controler = 0;
    pcard->current.location = LOCATION_MZONE;
    pcard->current.sequence = index;
    pcard->counters[counter_type] = capacities[index];
    fixture.game_field->player[0].list_mzone[index] = pcard;
  }
  fixture.game_field->select_counter(0, 0, counter_type, requested, 1, 0);
  std::vector<uint8_t> frame(fixture.message_buffer.begin(),
                             fixture.message_buffer.end());
  fixture.clear_buffer();
  ProtocolAdapterProbe probe;
  probe.attach(fixture, frame);
  std::vector<int> branch_sizes;
  for (int choice : choices) {
    if (probe.complete()) {
      throw std::runtime_error("Counter fixture has extra model choices");
    }
    branch_sizes.push_back(static_cast<int>(probe.choices()));
    probe.choose(choice);
  }
  if (!probe.complete()) {
    throw std::runtime_error("Incomplete adapter counter fixture sequence");
  }
  std::vector<uint16_t> actual;
  for (int index = 0; index < static_cast<int>(capacities.size()); ++index) {
    actual.push_back(fixture.game_field->returns.svalue[index]);
  }
  const bool accepted = fixture.game_field->select_counter(
      1, 0, counter_type, requested, 1, 0);
  return {frame, branch_sizes, actual, accepted};
}

inline std::tuple<std::vector<uint8_t>, std::vector<int>, bool>
core_weighted_adapter_fixture(CoreWeightedKind kind,
                              const std::vector<uint32_t> &must,
                              const std::vector<uint32_t> &optional,
                              int32_t target, int min_count, int max_count,
                              const std::vector<int> &choices,
                              bool cancelable = false) {
  if (optional.empty() || optional.size() > 32 || must.size() > 32 ||
      kind == CoreWeightedKind::Tribute && !must.empty()) {
    throw std::runtime_error("Invalid adapter weighted fixture size");
  }
  duel fixture;
  fixture.game_field->core.units.emplace_back();
  auto add_card = [&](uint32_t value, uint8_t sequence) {
    card *pcard = fixture.new_card(0);
    pcard->data.code = 1000 + sequence;
    pcard->current.controler = 0;
    pcard->current.location = LOCATION_HAND;
    pcard->current.sequence = sequence;
    pcard->release_param = value;
    pcard->sum_param = value;
    return pcard;
  };
  for (int index = 0; index < static_cast<int>(must.size()); ++index) {
    fixture.game_field->core.must_select_cards.push_back(
        add_card(must[index], static_cast<uint8_t>(index)));
  }
  for (int index = 0; index < static_cast<int>(optional.size()); ++index) {
    fixture.game_field->core.select_cards.push_back(add_card(
        optional[index], static_cast<uint8_t>(must.size() + index)));
  }
  if (kind == CoreWeightedKind::Tribute) {
    fixture.game_field->select_tribute(0, 0, cancelable,
                                      static_cast<uint8_t>(target),
                                      static_cast<uint8_t>(max_count));
  } else {
    fixture.game_field->select_with_sum_limit(
        0, 0, target, min_count,
        kind == CoreWeightedKind::ExactSum ? max_count : 0);
  }
  std::vector<uint8_t> frame(fixture.message_buffer.begin(),
                             fixture.message_buffer.end());
  fixture.clear_buffer();
  ProtocolAdapterProbe probe;
  probe.attach(fixture, frame);
  std::vector<int> branch_sizes;
  for (int choice : choices) {
    if (probe.complete()) {
      throw std::runtime_error("Weighted fixture has extra model choices");
    }
    branch_sizes.push_back(static_cast<int>(probe.choices()));
    probe.choose(choice);
  }
  if (!probe.complete()) {
    throw std::runtime_error("Incomplete adapter weighted fixture sequence");
  }
  bool accepted;
  if (kind == CoreWeightedKind::Tribute) {
    accepted = fixture.game_field->select_tribute(
        1, 0, cancelable, frame[3], frame[4]);
  } else {
    accepted = fixture.game_field->select_with_sum_limit(
        1, 0, target, min_count,
        kind == CoreWeightedKind::ExactSum ? max_count : 0);
  }
  return {frame, branch_sizes, accepted};
}

inline std::tuple<std::vector<uint8_t>, int, bool>
core_select_unselect_adapter_fixture(int select_count, int unselect_count,
                                     bool finishable, bool cancelable,
                                     int action_index) {
  if (select_count < 0 || unselect_count < 0 ||
      select_count + unselect_count < 1 ||
      select_count + unselect_count > 32) {
    throw std::runtime_error("Invalid adapter select/unselect fixture size");
  }
  duel fixture;
  auto add_card = [&](uint32_t code, uint8_t sequence) {
    card *pcard = fixture.new_card(0);
    pcard->data.code = code;
    pcard->current.controler = 0;
    pcard->current.location = LOCATION_HAND;
    pcard->current.sequence = sequence;
    pcard->current.position = POS_FACEUP;
    return pcard;
  };
  for (int index = 0; index < select_count; ++index) {
    fixture.game_field->core.select_cards.push_back(
        add_card(1000 + index, static_cast<uint8_t>(index)));
  }
  for (int index = 0; index < unselect_count; ++index) {
    fixture.game_field->core.unselect_cards.push_back(
        add_card(2000 + index, static_cast<uint8_t>(select_count + index)));
  }
  fixture.game_field->select_unselect_card(
      0, 0, cancelable, 1, 1, finishable);
  std::vector<uint8_t> frame(fixture.message_buffer.begin(),
                             fixture.message_buffer.end());
  fixture.clear_buffer();
  ProtocolAdapterProbe probe;
  probe.attach(fixture, frame);
  const int action_count = static_cast<int>(probe.choices());
  probe.choose(action_index);
  const bool accepted = fixture.game_field->select_unselect_card(
      1, 0, cancelable, 1, 1, finishable);
  return {frame, action_count, accepted};
}

inline std::tuple<std::vector<uint8_t>, std::vector<int>, bool>
core_select_card_adapter_fixture(int count, int min_count, int max_count,
                                 bool cancelable,
                                 const std::vector<int> &choices) {
  if (count < 1 || count > 32 || min_count < 0 ||
      max_count < 1 || max_count > count) {
    throw std::runtime_error("Invalid adapter select-card fixture shape");
  }
  duel fixture;
  fixture.game_field->core.units.emplace_back();
  for (int index = 0; index < count; ++index) {
    card *pcard = fixture.new_card(0);
    pcard->data.code = 1000 + index;
    pcard->current.controler = 0;
    pcard->current.location = LOCATION_HAND;
    pcard->current.sequence = index;
    pcard->current.position = POS_FACEUP;
    fixture.game_field->core.select_cards.push_back(pcard);
  }
  fixture.game_field->select_card(
      0, 0, cancelable, min_count, max_count);
  std::vector<uint8_t> frame(fixture.message_buffer.begin(),
                             fixture.message_buffer.end());
  fixture.clear_buffer();
  ProtocolAdapterProbe probe;
  probe.attach(fixture, frame);
  std::vector<int> branch_sizes;
  for (int choice : choices) {
    if (probe.complete()) {
      throw std::runtime_error("Select-card fixture has extra choices");
    }
    branch_sizes.push_back(static_cast<int>(probe.choices()));
    probe.choose(choice);
  }
  if (!probe.complete()) {
    throw std::runtime_error("Incomplete adapter select-card fixture");
  }
  const bool accepted = fixture.game_field->select_card(
      1, 0, cancelable, min_count, max_count);
  return {frame, branch_sizes, accepted};
}

inline std::tuple<std::vector<uint8_t>, int, uint32_t, bool>
core_mask_adapter_fixture(bool race, uint32_t available,
                          int count, int action_index) {
  duel fixture;
  fixture.game_field->core.units.emplace_back();
  if (race) fixture.game_field->announce_race(0, 0, count, available);
  else fixture.game_field->announce_attribute(0, 0, count, available);
  std::vector<uint8_t> frame(fixture.message_buffer.begin(),
                             fixture.message_buffer.end());
  fixture.clear_buffer();
  ProtocolAdapterProbe probe;
  probe.attach(fixture, frame);
  const int action_count = static_cast<int>(probe.choices());
  probe.choose(action_index);
  const uint32_t response = fixture.game_field->returns.ivalue[0];
  const bool accepted = race
      ? fixture.game_field->announce_race(1, 0, frame[2], available)
      : fixture.game_field->announce_attribute(1, 0, frame[2], available);
  return {frame, action_count, response, accepted};
}

inline std::tuple<std::vector<uint8_t>, int, uint32_t, bool>
core_number_adapter_fixture(const std::vector<uint32_t> &numbers,
                            int action_index) {
  if (numbers.empty() || numbers.size() > 32) {
    throw std::runtime_error("Invalid adapter number fixture size");
  }
  duel fixture;
  fixture.game_field->core.select_options.assign(numbers.begin(),
                                                 numbers.end());
  fixture.game_field->announce_number(0, 0);
  std::vector<uint8_t> frame(fixture.message_buffer.begin(),
                             fixture.message_buffer.end());
  fixture.clear_buffer();
  ProtocolAdapterProbe probe;
  probe.attach(fixture, frame);
  const int action_count = static_cast<int>(probe.choices());
  probe.choose(action_index);
  const uint32_t response = fixture.game_field->returns.ivalue[0];
  const bool accepted = fixture.game_field->announce_number(1, 0);
  return {frame, action_count, response, accepted};
}

inline std::tuple<std::vector<uint8_t>, int, uint32_t, bool>
core_position_adapter_fixture(uint8_t positions, int action_index) {
  duel fixture;
  const uint32_t code = 1000;
  const auto previous = card_ids_.find(code);
  const bool inserted = previous == card_ids_.end();
  if (inserted) card_ids_[code] = 1;
  try {
    fixture.game_field->select_position(0, 0, code, positions);
    std::vector<uint8_t> frame(fixture.message_buffer.begin(),
                               fixture.message_buffer.end());
    fixture.clear_buffer();
    ProtocolAdapterProbe probe;
    probe.attach(fixture, frame);
    const int action_count = static_cast<int>(probe.choices());
    probe.choose(action_index);
    const uint32_t response = fixture.game_field->returns.ivalue[0];
    const bool accepted = fixture.game_field->select_position(
        1, 0, code, positions);
    if (inserted) card_ids_.erase(code);
    return {frame, action_count, response, accepted};
  } catch (...) {
    if (inserted) card_ids_.erase(code);
    throw;
  }
}

inline std::tuple<std::vector<uint8_t>, std::vector<int>, bool>
core_place_adapter_fixture(bool disfield, uint32_t disabled,
                           uint8_t count, const std::vector<int> &choices) {
  duel fixture;
  fixture.game_field->core.units.emplace_back();
  fixture.game_field->core.units.begin()->type =
      disfield ? PROCESSOR_SELECT_DISFIELD : PROCESSOR_SELECT_PLACE;
  fixture.game_field->select_place(0, 0, disabled, count);
  std::vector<uint8_t> frame(fixture.message_buffer.begin(),
                             fixture.message_buffer.end());
  fixture.clear_buffer();
  ProtocolAdapterProbe probe;
  probe.attach(fixture, frame);
  std::vector<int> branch_sizes;
  for (int choice : choices) {
    if (probe.complete()) {
      throw std::runtime_error("Place fixture has extra choices");
    }
    branch_sizes.push_back(static_cast<int>(probe.choices()));
    probe.choose(choice);
  }
  if (!probe.complete()) {
    throw std::runtime_error("Incomplete adapter place fixture");
  }
  const bool accepted = fixture.game_field->select_place(
      1, 0, disabled, count);
  return {frame, branch_sizes, accepted};
}

inline std::tuple<std::vector<uint8_t>, int, int32_t, bool>
core_scalar_adapter_fixture(int kind, int action_index) {
  if (kind < 0 || kind > 3) {
    throw std::runtime_error("Invalid scalar adapter fixture kind");
  }
  duel fixture;
  card *pcard = nullptr;
  if (kind == 0) fixture.game_field->select_yes_no(0, 0, 1);
  else if (kind == 1) {
    pcard = fixture.new_card(0);
    pcard->data.code = 1000;
    pcard->current.controler = 0;
    pcard->current.location = LOCATION_HAND;
    pcard->current.sequence = 0;
    pcard->current.position = POS_FACEUP;
    fixture.game_field->select_effect_yes_no(0, 0, 1, pcard);
  } else {
    fixture.game_field->core.select_options =
        kind == 3 ? std::vector<uint32_t>{1}
                  : std::vector<uint32_t>{1, 2, 3};
    fixture.game_field->select_option(0, 0);
  }
  std::vector<uint8_t> frame(fixture.message_buffer.begin(),
                             fixture.message_buffer.end());
  fixture.clear_buffer();
  ProtocolAdapterProbe probe;
  probe.attach(fixture, frame);
  const int action_count = static_cast<int>(probe.choices());
  probe.choose(action_index);
  const int32_t response = fixture.game_field->returns.ivalue[0];
  const bool accepted = kind == 0
      ? fixture.game_field->select_yes_no(1, 0, 1)
      : kind == 1
          ? fixture.game_field->select_effect_yes_no(1, 0, 1, pcard)
          : fixture.game_field->select_option(1, 0);
  return {frame, action_count, response, accepted};
}

inline std::tuple<std::vector<uint8_t>, int, int32_t, bool>
core_command_adapter_fixture(bool battle, int action_index) {
  duel fixture;
  field *game = fixture.game_field;
  card *pcard = fixture.new_card(0);
  pcard->data.code = 0;
  pcard->current.controler = 0;
  pcard->current.location = LOCATION_MZONE;
  pcard->current.sequence = 0;
  effect *peffect = fixture.new_effect();
  peffect->handler = pcard;
  peffect->description = 0;
  chain ch;
  ch.triggering_effect = peffect;
  game->core.select_chains.push_back(ch);
  if (battle) {
    game->core.attackable_cards.push_back(pcard);
    game->core.to_m2 = 1;
    game->core.to_ep = 1;
    game->select_battle_command(0, 0);
  } else {
    game->core.summonable_cards.push_back(pcard);
    game->core.spsummonable_cards.push_back(pcard);
    game->core.repositionable_cards.push_back(pcard);
    game->core.msetable_cards.push_back(pcard);
    game->core.ssetable_cards.push_back(pcard);
    game->core.to_bp = 1;
    game->core.to_ep = 1;
    game->infos.phase = PHASE_MAIN1;
    game->infos.can_shuffle = 1;
    game->player[0].list_hand.push_back(pcard);
    game->player[0].list_hand.push_back(pcard);
    game->select_idle_command(0, 0);
  }
  std::vector<uint8_t> frame(fixture.message_buffer.begin(),
                             fixture.message_buffer.end());
  fixture.clear_buffer();
  ProtocolAdapterProbe probe;
  probe.attach(fixture, frame);
  probe.assert_frozen_command_domains();
  const int action_count = static_cast<int>(probe.choices());
  probe.choose(action_index);
  const int32_t response = game->returns.ivalue[0];
  const bool accepted = battle ? game->select_battle_command(1, 0)
                               : game->select_idle_command(1, 0);
  return {frame, action_count, response, accepted};
}

inline std::tuple<std::vector<uint8_t>, int, int32_t, bool>
core_chain_adapter_fixture(int count, bool forced, int action_index) {
  if (count < 0 || count > 8 || (forced && count == 0)) {
    throw std::runtime_error("Invalid chain adapter fixture count");
  }
  duel fixture;
  field *game = fixture.game_field;
  for (int index = 0; index < count; ++index) {
    card *pcard = fixture.new_card(0);
    pcard->data.code = 0;
    pcard->current.controler = 0;
    pcard->current.location = LOCATION_MZONE;
    pcard->current.sequence = index;
    effect *peffect = fixture.new_effect();
    peffect->handler = pcard;
    peffect->description = 0;
    chain ch;
    ch.triggering_effect = peffect;
    game->core.select_chains.push_back(ch);
  }
  game->select_chain(0, 0, 0, forced);
  std::vector<uint8_t> frame(fixture.message_buffer.begin(),
                             fixture.message_buffer.end());
  fixture.clear_buffer();
  ProtocolAdapterProbe probe;
  probe.attach(fixture, frame);
  const int action_count = static_cast<int>(probe.choices());
  if (action_index >= 0) probe.choose(action_index);
  const int32_t response = game->returns.ivalue[0];
  const bool accepted = game->select_chain(1, 0, 0, forced);
  return {frame, action_count, response, accepted};
}

inline uint32_t protocol_fixture_card_reader(uint32_t code, card_data *data) {
  if (code >= 23456789 && code < 23457089 && cards_data_.count(code)) {
    *data = cards_data_.at(code);
    return 0;
  }
  return default_card_reader(code, data);
}

inline std::tuple<std::vector<uint8_t>, int, int32_t, bool>
core_announce_card_adapter_fixture(bool named, int action_index) {
  constexpr uint32_t first = 23456789;
  constexpr uint32_t second = 23456790;
  if (cards_data_.count(first) || cards_data_.count(second) ||
      card_ids_.count(first) || card_ids_.count(second)) {
    throw std::runtime_error("Announce-card fixture code already registered");
  }
  for (uint32_t code : {first, second}) {
    card_data data{};
    data.code = code;
    data.type = TYPE_MONSTER;
    cards_data_[code] = data;
    card_ids_[code] = static_cast<CardId>(code - first + 1);
  }
  set_card_reader(protocol_fixture_card_reader);
  try {
    duel fixture;
    field *game = fixture.game_field;
    game->core.select_options = named
        ? std::vector<uint32_t>{second, OPCODE_ISCODE,
                                first, OPCODE_ISCODE, OPCODE_OR}
        : std::vector<uint32_t>{TYPE_MONSTER, OPCODE_ISTYPE,
                                TYPE_TOKEN, OPCODE_ISTYPE,
                                OPCODE_NOT, OPCODE_AND};
    game->announce_card(0, 0);
    std::vector<uint8_t> frame(fixture.message_buffer.begin(),
                               fixture.message_buffer.end());
    fixture.clear_buffer();
    ProtocolAdapterProbe probe;
    probe.attach(fixture, frame);
    const int action_count = static_cast<int>(probe.choices());
    probe.choose(action_index);
    const int32_t response = game->returns.ivalue[0];
    const bool accepted = game->announce_card(1, 0);
    cards_data_.erase(first); cards_data_.erase(second);
    card_ids_.erase(first); card_ids_.erase(second);
    set_card_reader(default_card_reader);
    return {frame, action_count, response, accepted};
  } catch (...) {
    cards_data_.erase(first); cards_data_.erase(second);
    card_ids_.erase(first); card_ids_.erase(second);
    set_card_reader(default_card_reader);
    throw;
  }
}

inline std::tuple<std::vector<uint8_t>, std::vector<int>, int32_t, bool>
core_announce_card_large_adapter_fixture(const std::vector<int> &choices) {
  constexpr uint32_t first = 23456789;
  constexpr int count = 300;
  for (int index = 0; index < count; ++index) {
    const uint32_t code = first + index;
    if (cards_data_.count(code) || card_ids_.count(code)) {
      throw std::runtime_error("Large announce-card fixture code registered");
    }
    card_data data{};
    data.code = code;
    data.type = TYPE_MONSTER;
    cards_data_[code] = data;
    card_ids_[code] = index + 1;
  }
  auto cleanup = []() {
    for (int index = 0; index < count; ++index) {
      cards_data_.erase(first + index);
      card_ids_.erase(first + index);
    }
    set_card_reader(default_card_reader);
  };
  set_card_reader(protocol_fixture_card_reader);
  try {
    duel fixture;
    field *game = fixture.game_field;
    game->core.select_options = {TYPE_MONSTER, OPCODE_ISTYPE};
    game->announce_card(0, 0);
    std::vector<uint8_t> frame(fixture.message_buffer.begin(),
                               fixture.message_buffer.end());
    fixture.clear_buffer();
    ProtocolAdapterProbe probe;
    probe.attach(fixture, frame);
    std::vector<int> branch_sizes;
    for (int choice : choices) {
      branch_sizes.push_back(static_cast<int>(probe.choices()));
      probe.choose(choice);
    }
    if (!probe.complete()) {
      throw std::runtime_error("Incomplete staged announce-card choice");
    }
    const int32_t response = game->returns.ivalue[0];
    const bool accepted = game->announce_card(1, 0);
    cleanup();
    return {frame, branch_sizes, response, accepted};
  } catch (...) {
    cleanup();
    throw;
  }
}

inline std::pair<std::vector<std::vector<uint8_t>>, int32_t>
core_rps_adapter_fixture(const std::vector<int> &hands, bool repeat) {
  duel fixture;
  fixture.game_field->add_process(PROCESSOR_ROCK_PAPER_SCISSORS, 0, nullptr,
                                  nullptr, repeat ? 1 : 0, 0);
  ProtocolAdapterProbe probe;
  std::vector<std::vector<uint8_t>> frames;
  size_t hand_index = 0;
  for (size_t step = 0; step < 128; ++step) {
    fixture.clear_buffer();
    fixture.game_field->process();
    if (fixture.message_buffer.empty()) break;
    std::vector<uint8_t> frame(fixture.message_buffer.begin(),
                               fixture.message_buffer.end());
    frames.push_back(frame);
    if (frame[0] == MSG_ROCK_PAPER_SCISSORS) {
      if (hand_index >= hands.size() || hands[hand_index] < 1 ||
          hands[hand_index] > 3) {
        throw std::runtime_error("Invalid RPS adapter fixture hand");
      }
      probe.attach(fixture, frame);
      if (probe.choices() != 3) {
        throw std::runtime_error("RPS adapter did not expose all hands");
      }
      probe.choose(hands[hand_index++] - 1);
    } else if (frame[0] == MSG_HAND_RES &&
               fixture.game_field->core.units.empty()) {
      if (hand_index != hands.size()) {
        throw std::runtime_error("Unused RPS adapter fixture hands");
      }
      return {frames, fixture.game_field->returns.ivalue[0]};
    }
  }
  throw std::runtime_error("RPS adapter did not reach terminal result");
}

class YGOProEnv : public Env<YGOProEnvSpec> {
protected:
  const int max_episode_steps_;
  const int timeout_;

  int elapsed_step_;

  std::uniform_int_distribution<uint64_t> dist_int_;

  // The pool can't be in vector, so we create multiple pools manually
  BS::thread_pool pool0_;
  BS::thread_pool pool1_;
  BS::thread_pool pool2_;
  BS::thread_pool pool3_;
  BS::thread_pool pool4_;

  const int max_timeout_{5};
 
  // YGOProEnvImpl env_impl0_;
  // YGOProEnvImpl env_impl1_;
  // YGOProEnvImpl env_impl2_;
  // YGOProEnvImpl env_impl3_;
  // YGOProEnvImpl env_impl4_;
  std::vector<YGOProEnvImpl> env_impls_;

  bool done_{true};

public:
  YGOProEnv(const Spec &spec, int env_id)
      : Env<YGOProEnvSpec>(spec, env_id),
        max_episode_steps_(spec.config["max_episode_steps"_]),
        elapsed_step_(max_episode_steps_ + 1),
        timeout_(spec.config["timeout"_]),
        pool0_(1), pool1_(1), pool2_(1), pool3_(1), pool4_(1),
        dist_int_(0, 0xffffffff) {
    env_impls_.reserve(max_timeout_);
    env_impls_.emplace_back(spec, dist_int_(gen_), env_id_);
  }

  bool IsDone() override { return done_; }

  BS::thread_pool& get_pool(int idx) {
    switch (idx) {
      case 0: return pool0_;
      case 1: return pool1_;
      case 2: return pool2_;
      case 3: return pool3_;
      case 4: return pool4_;
      default: throw std::runtime_error("Invalid pool index");
    }
  }

  void handle_timeout() {
    const int current = static_cast<int>(env_impls_.size()) - 1;
    fmt::println("Env {} timeout: {}", env_id_,
                 env_impls_[current].timeout_diagnostic());
    if (env_impls_.size() >= static_cast<size_t>(max_timeout_)) {
      throw std::runtime_error("Too many timeouts");
    }
    env_impls_.emplace_back(spec_, dist_int_(gen_), env_id_);
    done_ = true;
    State state = Allocate();
    state["reward"_] = 0.0;
    state["info:to_play"_] = 1;
    state["info:is_selfplay"_] = 1;
    state["info:win_reason"_] = 1;
    state["info:num_options"_] = 1;
    state["info:invalid_game"_] = 1;
    state["info:termination_reason"_] = int(kTerminationTimeout);
    state["info:episode_steps"_] = 0;
    state["info:turn_count"_] = 0;
    state["obs:global_"_][22] = uint8_t(1);
  }

  void Reset() override {
    int idx = env_impls_.size() - 1;
    auto& pool = get_pool(idx);
    auto fut = pool.submit_task([this, idx]() {
      env_impls_[idx].reset();
    });
    if (fut.wait_for(std::chrono::seconds(timeout_)) != std::future_status::ready) {
      throw std::runtime_error("Reset timeout");
    }

    auto &env_impl = env_impls_[idx];
    elapsed_step_ = 0;
    done_ = false;
    State state = Allocate();
    env_impl.WriteState(state);
  }

  void Step(const Action &action) override {
    int idx = env_impls_.size() - 1;
    auto& pool = get_pool(idx);
    int action_idx = action["action"_];
    pool.detach_task([this, action_idx, idx]() {
      // Test timeout: random sleep with probability 0.01
      // if (dist_int_(gen_) % 10000 == 0) {
      //   fmt::println("Env {} sleep {}", env_id_, env_impls_.capacity());
      //   std::this_thread::sleep_for(std::chrono::seconds(5));
      //   fmt::println("Env {} after {}", env_id_, env_impls_.capacity());
      //   auto& env_impl = env_impls_[idx];
      //   env_impl.step(action_idx);
      //   std::this_thread::sleep_for(std::chrono::seconds(1));
      //   return;
      // }
      env_impls_[idx].step(action_idx);
    });
    if (!pool.wait_for(std::chrono::seconds(timeout_))) {
      handle_timeout();
      fmt::println("Env {} timeout, new env created", env_id_);
    } else {
      auto& env_impl = env_impls_[idx];
      done_ = env_impl.is_done();
      State state = Allocate();
      env_impl.WriteState(state);
    }
  }

};

using YGOProEnvPool = AsyncEnvPool<YGOProEnv>;

} // namespace ygopro

template <>
struct fmt::formatter<ygopro::LegalAction>: formatter<string_view> {

    // Format the LegalAction object
    template <typename FormatContext>
    auto format(const ygopro::LegalAction& action, FormatContext& ctx) const {
        std::stringstream ss;
        ss << "{";
        if (!action.spec_.empty()) {
          ss << "spec='" << action.spec_ << "', ";
        }
        if (action.cid_ != 0) {
          ss << "cid=" << action.cid_ << ", ";
        }
        if (action.act_ != ygopro::ActionAct::None) {
          ss << "act=" << ygopro::action_act_to_string(action.act_) << ", ";
        }
        if (action.phase_ != ygopro::ActionPhase::None) {
          ss << "phase=" << ygopro::action_phase_to_string(action.phase_) << ", ";
        }
        if (action.finish_) {
          ss << "finish=true, ";
        }
        if (action.position_ != 0) {
          ss << "position=" << ygopro::position_to_string(action.position_) << ", ";
        }
        if (action.effect_ != -1) {
          ss << "effect=" << action.effect_ << ", ";
        }
        if (action.number_ != 0) {
          ss << "number=" << int(action.number_) << ", ";
        }
        if (action.place_ != ygopro::ActionPlace::None) {
          ss << "place=" << ygopro::action_place_to_string(action.place_) << ", ";
        }
        if (action.attribute_ != 0) {
          ss << "attribute=" << ygopro::attribute_to_string(action.attribute_) << ", ";
        }
        if (action.race_ != 0) {
          ss << "race_mask=0x" << std::hex << action.race_ << std::dec << ", ";
        }
        std::string s = ss.str();
        if (s.back() == ' ') {
          s.pop_back();
          s.pop_back();
        }
        s.push_back('}');
        return format_to(ctx.out(), "{}", s);
    }
};

#endif // YGOENV_YGOPRO_YGOPRO_H_
