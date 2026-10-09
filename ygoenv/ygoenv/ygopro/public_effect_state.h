#ifndef YGO_PUBLIC_EFFECT_STATE_H_
#define YGO_PUBLIC_EFFECT_STATE_H_
#include <array>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <tuple>

namespace ygopro {
// Only MSG_CHAINING and subsequent public chain messages enter this ledger.
// Counts aggregate copies of a card controlled by the same seat; they are NOT
// physical-card counters, OPT availability, or successful-effect labels.
class PublicEffectState {
 public:
  using Key = std::tuple<uint8_t, uint32_t, uint8_t>;
  struct Link { Key key; uint16_t status = 1; };
  std::map<uint8_t, Link> chain;
  std::map<Key, std::array<uint16_t, 4>> turn;

  void reset() { chain.clear(); turn.clear(); }
  void end_chain() { chain.clear(); }
  static uint8_t description_slot(uint32_t code, uint32_t description) {
    return code != 0 && (description >> 4) == code
        ? static_cast<uint8_t>((description & 15) + 1) : 0;
  }
  static void increment(uint16_t &count) {
    if (count == 65535) throw std::runtime_error("public effect count overflow");
    ++count;
  }
  void activate(uint32_t code, uint8_t controller, uint32_t description, uint8_t link) {
    if (!code || controller > 1 || !link || link > 16)
      throw std::runtime_error("public activation outside state contract");
    if (link == 1) chain.clear();
    if (chain.count(link)) throw std::runtime_error("duplicate public chain link");
    Key key{controller, code, description_slot(code, description)};
    if (!turn.count(key) && turn.size() >= 64)
      throw std::runtime_error("public turn effect capacity exceeded");
    increment(turn[key][0]);
    chain.emplace(link, Link{key, 1});
  }
  void update(uint8_t link, uint16_t flag) {
    auto it = chain.find(link);
    if (it == chain.end()) throw std::runtime_error("public chain link has no activation");
    auto &entry = it->second;
    if (entry.status & flag) return;  // Count each status once per activation.
    entry.status |= flag;
    if (flag == 4) increment(turn.at(entry.key)[1]);
    if (flag == 8) increment(turn.at(entry.key)[2]);
    if (flag == 16) increment(turn.at(entry.key)[3]);
  }
};
}  // namespace ygopro
#endif
