"""Prepare an isolated normal factory from clean archived source.

The count fix is already in source. Only natural-draw settlement changes here;
no nonterminal observation or legal action changes. Never edit a live module.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('missing or ambiguous source anchor: ' + old)
    return text.replace(old, new)


def prepare(root, output):
    header = root/'ygoenv/ygoenv/ygopro/ygopro.h'
    binding = root/'ygoenv/ygoenv/ygopro/ygopro.cpp'
    original = sha(header)
    text = header.read_text()
    assert 'unselect_counts_' in text and 'if (play_mode_ == kHuman || full_legal_menu_) return;' in text
    text = replace_once(text, '      if (!invalid_game_) {',
                        '      if (!invalid_game_ && winner_ != 2) {')
    text = replace_once(text, '  ProtocolAdapterProbe() : YGOProEnvImpl(YGOProEnvSpec{}, 0) {',
                        '''  ProtocolAdapterProbe(bool greedy = false) : YGOProEnvImpl(YGOProEnvSpec([greedy]() {
    auto config = YGOProEnvSpec::kDefaultConfig;
    config["greedy_reward"_] = greedy;
    return config;
  }()), 0) {''')
    text = replace_once(text, '''      auto& winner = players_[player];
      auto& loser = players_[1 - player];

      _duel_end(player, reason);

      auto l_reason = reason_to_string(reason);
      if (verbose_) {
        winner->notify("You won (" + l_reason + ").");
        loser->notify("You lost (" + l_reason + ").");
      }''', '''      if (player > 2) throw std::runtime_error("Invalid duel winner");
      _duel_end(player, reason);
      auto l_reason = reason_to_string(reason);
      if (verbose_) {
        if (player == 2) {
          for (auto& participant : players_) participant->notify("Draw (" + l_reason + ").");
        } else {
          players_[player]->notify("You won (" + l_reason + ").");
          players_[1 - player]->notify("You lost (" + l_reason + ").");
        }
      }''')
    # Probe uses the actual parser, step() and reward code; no duplicated formula.
    text = replace_once(text, '  size_t choices() const { return legal_actions_.size(); }', '''  size_t choices() const { return legal_actions_.size(); }

  std::tuple<int, float, bool, bool> reality_terminal_fixture(
      uint8_t winner, int seat, bool greedy, bool verbose) {
    pduel_ = YGO_CreateDuel(0);
    duel_started_ = true;
    play_mode_ = kSelfPlay;
    to_play_ = seat;
    verbose_ = verbose;
    invalid_game_ = false;
    done_ = false;
    turn_count_ = 4;
    current_phase_ = PHASE_MAIN1;
    try {
      attach(*reinterpret_cast<duel*>(pduel_), {MSG_WIN, winner, 1});
      LegalAction action = LegalAction::cancel();
      action.msg_ = MSG_SELECT_CHAIN;
      legal_actions_ = {action};
      callback_ = [](int) {};
      ms_idx_ = -1;
      step(0);
    } catch (...) {
      if (duel_started_) YGO_EndDuel(pduel_);
      duel_started_ = false;
      throw;
    }
    return {int(winner_), ret_reward_, done_, duel_started_};
  }''')
    header.write_text(text)
    text = binding.read_text()
    attributes = '''
  m.attr("exercise_training_scope_version") = "selection-counts-contextual-navigation-draw-v2";
  m.attr("exercise_training_header_sha256") = "HEADER_HASH";
  m.def("_reality_terminal_fixture", [](uint8_t winner, int seat, bool greedy, bool verbose) {
    ygopro::ProtocolAdapterProbe probe(greedy);
    return probe.reality_terminal_fixture(winner, seat, greedy, verbose);
  });
  m.def("_reality_navigation_filter", [](int message, bool summon_navigation) {
    ygopro::ProtocolAdapterProbe probe;
    return probe.policy_cancel_filter_fixture(message, summon_navigation);
  });'''.replace('HEADER_HASH', sha(header))
    binding.write_text(replace_once(text, 'PYBIND11_MODULE(ygopro_ygoenv, m) {',
                                   'PYBIND11_MODULE(ygopro_ygoenv, m) {' + attributes))
    metadata = dict(source_header_sha256=original, header_sha256=sha(header),
                    binding_sha256=sha(binding), scope='normal factory; selection counts from source; isolated terminal draw repair and real parser/step probe',
                    nonterminal_observations_changed=False, legal_actions_changed=False,
                    policy_full_legal_menu=False, production_replaced=False)
    (output/'source-binding.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    prepare(a.root, a.output)
