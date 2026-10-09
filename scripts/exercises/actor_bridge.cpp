// Diagnostic subclass of the actual adapter; policy sees its unchanged writer.
#include "ocgapi.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
namespace py = pybind11;
static std::vector<std::string> packets;
static std::vector<std::string> responses;
static std::vector<bool> response_is_int;
static std::vector<std::vector<uint8_t>> last_fields;
static int process_calls = 0;
static void capture_fields(intptr_t d) {
  last_fields.clear();
  for (int p = 0; p < 2; ++p) for (int loc : {1,2,4,8,16,32,64}) {
    std::vector<uint8_t> b(65536);
    int n = query_field_card(d, p, loc, 3, b.data(), 0);
    b.resize(n); last_fields.push_back(std::move(b));
  }
}
static int exercise_get_message(intptr_t d, uint8_t* b) {
  int n = get_message(d, b);
  packets.emplace_back(reinterpret_cast<char*>(b), n);
  capture_fields(d); // before the original adapter releases a terminal duel
  return n;
}
static uint32_t exercise_process(intptr_t d) {
  if (++process_calls > 4096) throw std::runtime_error("exercise process budget exhausted");
  return process(d);
}
static void exercise_responsei(intptr_t d, int32_t value) {
  responses.emplace_back(reinterpret_cast<char*>(&value), 4);
  response_is_int.push_back(true);
  set_responsei(d, value);
}
static void exercise_responseb(intptr_t d, uint8_t* b, int msg, int places) {
  int n = (msg == 18 || msg == 24) ? 3 * places : b[0] + 1;
  if (n < 1 || n > 64) throw std::runtime_error("unsupported exercise byte response");
  responses.emplace_back(reinterpret_cast<char*>(b), n);
  response_is_int.push_back(false);
  set_responseb(d, b);
}
#include "exercise_ygopro.h"

namespace ygopro {
static YGOProEnvSpec exercise_spec(const std::string& semantics, const std::string& schema) {
  auto conf = YGOProEnvSpec::kDefaultConfig;
  conf["play_mode"_] = std::string("self");
  conf["deck1"_] = conf["deck2"_] = std::string("_exercise");
  conf["max_options"_] = 128;
  conf["n_history_actions"_] = 32;
  conf["max_steps"_] = 160;
  conf["observation_schema"_] = schema;
  conf["semantic_asset_dir"_] = semantics;
  conf["async_reset"_] = false;
  conf["greedy_reward"_] = false;
  return YGOProEnvSpec(conf);
}

class ExerciseActor : public YGOProEnvImpl {
  std::string lua_;
  std::vector<std::vector<uint32_t>> initial_;
  uint32_t core_seed_ = 0;
public:
  ExerciseActor(const std::string& lua, const std::vector<std::vector<uint32_t>>& initial,
                const std::string& semantics, int seed, const std::string& schema)
      : YGOProEnvImpl(exercise_spec(semantics, schema), seed), lua_(lua), initial_(initial) {
    ai_player_ = 0; // not used by self-play, but reset uses it for nicknames
  }
  ~ExerciseActor() { close(); }
  void close() {
    if (duel_started_) { delete reinterpret_cast<duel*>(pduel_); duel_started_ = false; }
  }
  MDuel new_duel(uint32_t seed) override {
    auto* pd = new duel();
    std::mt19937 rnd(seed);
    core_seed_ = rnd();
    pd->random.reset(core_seed_); // same factory convention as YGO_CreateDuel
    auto d = reinterpret_cast<intptr_t>(pd);
    MDuel out{d, seed};
    out.deck_name0 = out.deck_name1 = "_exercise";
    if (lua_.empty()) {
      for (int p = 0; p < 2; ++p) set_player_info(d, p, 8000, 5, 1);
    }
    for (auto& row : initial_) {
      if (row.size() != 5) throw std::runtime_error("bad initial card record");
      if (lua_.empty()) {
        if (row[1] > 1 || (row[2] != 1 && row[2] != 64))
          throw std::runtime_error("standard opening permits decks only");
        new_card(d, row[0], row[1], row[1], row[2], row[3], row[4]);
      }
      auto& target = row[1] == 0
        ? (row[2] == 64 ? out.extra_deck0 : out.main_deck0)
        : (row[2] == 64 ? out.extra_deck1 : out.main_deck1);
      target.push_back(row[0]);
    }
    if (lua_.empty()) {
      start_duel(d, 4 << 16); // standard draw, first-turn restrictions; fixed legal deal
      return out;
    }
    const std::string name = "./script/__exercise_actor.lua";
    cards_script_[name] = {reinterpret_cast<byte*>(lua_.data()), int(lua_.size())};
    if (!preload_script(d, name.c_str(), name.size())) {
      delete pd;
      throw std::runtime_error("exercise preload failed");
    }
    start_duel(d, DUEL_ATTACK_FIRST_TURN | (4 << 16));
    return out;
  }
  void begin() {
    packets.clear(); responses.clear(); response_is_int.clear(); process_calls = 0;
    reset(); // original initialization, original parser and forced-action path
  }
  py::dict snapshot() {
    State state;
    std::apply([&](auto... key) {
      ([&] {
        ShapeSpec shape = spec_.state_spec[key];
        for (int& dim : shape.shape) if (dim < 0) dim = 1;
        state[key] = std::decay_t<decltype(state[key])>(Array(shape));
        state[key].Zero();
      }(), ...);
    }, State::StaticKeys());
    WriteState(state);
    py::dict obs;
    std::apply([&](auto... key) {
      ([&] {
        std::string name = key.Str();
        if (name.rfind("obs:", 0) != 0) return;
        auto& a = state[key];
        std::vector<py::ssize_t> shape{1};
        for (auto n : a.Shape()) shape.push_back(n);
        auto dtype = a.element_size == 2 ? py::dtype::of<uint16_t>() : py::dtype::of<uint8_t>();
        py::array copied(dtype, shape);
        std::memcpy(copied.mutable_data(), a.Data(), a.size * a.element_size);
        obs[py::str(name.substr(4))] = copied;
      }(), ...);
    }, State::StaticKeys());
    py::list menu;
    for (const auto& a : legal_actions_) {
      py::dict item;
      uint32_t code = 0;
      for (const auto& pair : card_ids_) if (pair.second == a.cid_) { code = pair.first; break; }
      item["code"] = code; item["spec"] = a.spec_;
      item["act"] = action_act_to_string(a.act_);
      item["phase"] = action_phase_to_string(a.phase_);
      item["finish"] = a.finish_; item["position"] = a.position_;
      item["effect"] = a.effect_; item["unselect"] = a.unselect_;
      item["place"] = int(a.place_); item["response"] = a.response_;
      menu.append(item);
    }
    py::dict result;
    result["observation"] = obs; result["menu"] = menu;
    result["player"] = int(to_play_); result["message"] = msg_;
    result["done"] = done_; result["invalid"] = invalid_game_;
    result["phase"] = current_phase_; result["turn_player"] = int(tp_);
    result["winner"] = done_ && !invalid_game_ ? int(winner_) : -1;
    result["lp"] = std::vector<int>{lp_[0], lp_[1]};
    result["process_calls"] = process_calls;
    return result;
  }
  py::dict trace() {
    if (duel_started_) capture_fields(pduel_);
    py::dict result;
    py::list ps, rs, fs;
    for (auto& p : packets) ps.append(py::bytes(p));
    for (size_t i = 0; i < responses.size(); ++i) {
      py::dict r; r["kind"] = response_is_int[i] ? "int" : "bytes";
      r["data"] = py::bytes(responses[i]); rs.append(r);
    }
    for (auto& f : last_fields) fs.append(py::bytes(reinterpret_cast<char*>(f.data()), f.size()));
    result["packets"] = ps; result["responses"] = rs; result["fields"] = fs;
    result["core_seed"] = core_seed_;
    return result;
  }
};
}

PYBIND11_MODULE(exercise_actor_native, m) {
  m.def("init_module", &ygopro::init_module);
  py::class_<ygopro::ExerciseActor>(m, "ExerciseActor")
    .def(py::init<const std::string&, const std::vector<std::vector<uint32_t>>&, const std::string&, int, const std::string&>(),
         py::arg("lua"), py::arg("initial"), py::arg("semantics"), py::arg("seed") = 1,
         py::arg("observation_schema") = "structured-lite-v1")
    .def("reset", &ygopro::ExerciseActor::begin)
    .def("step", &ygopro::ExerciseActor::step)
    .def("snapshot", &ygopro::ExerciseActor::snapshot)
    .def("trace", &ygopro::ExerciseActor::trace)
    .def("close", &ygopro::ExerciseActor::close);
}
