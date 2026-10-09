#include "ygoenv/ygopro/public_effect_state.h"
#include <cassert>
#include <functional>
using ygopro::PublicEffectState;
bool rejects(const std::function<void()> &f) { try { f(); return false; } catch (const std::runtime_error &) { return true; } }
int main() {
  PublicEffectState s;
  s.activate(26077387, 0, (26077387u << 4), 1);
  s.activate(97268402, 1, (97268402u << 4) | 1, 2);
  s.update(1, 8); s.update(2, 16); s.update(2, 2); s.update(2, 4); s.update(1, 4);
  assert(s.chain.at(1).status == 13);
  assert(s.chain.at(2).status == 23);
  assert(s.turn.at({0,26077387,1})[2] == 1);
  assert(s.turn.at({1,97268402,2})[3] == 1);
  s.update(2,16); assert(s.turn.at({1,97268402,2})[3] == 1);
  s.end_chain(); assert(s.chain.empty()); assert(s.turn.size() == 2);
  // Counts survive event-window truncation and many later chains.
  for(int i=0;i<40;++i) { s.activate(26077387,0,26077387u<<4,1); s.update(1,4); s.end_chain(); }
  assert(s.turn.at({0,26077387,1})[0] == 41);
  assert(rejects([&]{ s.update(1,4); }));
  s.reset(); assert(s.turn.empty()); assert(s.chain.empty());
  s.activate(26077387,0,1234,1); assert(std::get<2>(s.chain.at(1).key) == 0);
  assert(rejects([&]{ s.activate(1,0,16,17); }));
  s.reset();
  for(unsigned i=1;i<=64;++i) { s.activate(i,0,i<<4,1); s.end_chain(); }
  assert(rejects([&]{ s.activate(65,0,65<<4,1); }));
}
