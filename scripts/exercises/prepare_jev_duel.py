"""Generate an isolated full-duel bridge around the existing legal-action adapter.

Never edits the production header/module. Use a clean source archive as input.
"""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.exercises.prepare_actor_header import prepare


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('missing/ambiguous bridge anchor: '+old)
    return text.replace(old, new)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    header = a.output/'exercise_ygopro.h'
    prepare('ygoenv/ygoenv/ygopro/ygopro.h', header)
    h = header.read_text()
    h = once(h, 'exercise_responseb(pduel, buf, msg_, n_places_);',
             'exercise_responseb(pduel, buf, msg_, n_places_, n_sort_cards_, n_counters_);')
    h = once(h, '  void handle_message() {',
             '  void handle_message() {\n    JevFrameCapture jev_capture(data_, dp_);')
    header.write_text(h)
    s = Path('scripts/exercises/actor_bridge.cpp').read_text()
    s = once(s, 'static int process_calls = 0;', '''static int process_calls = 0;
static std::vector<std::string> jev_frames;
struct JevFrameCapture {
  const uint8_t* data; const int& end; int start;
  JevFrameCapture(const uint8_t* d, const int& e): data(d), end(e), start(e) {}
  ~JevFrameCapture() { if (end > start) jev_frames.emplace_back(reinterpret_cast<const char*>(data+start), end-start); }
};''')
    s = once(s, 'packets.clear(); responses.clear();', 'jev_frames.clear(); packets.clear(); responses.clear();')
    s = once(s, 'result["packets"] = ps;', '''py::list frames;
    for (auto& f : jev_frames) frames.append(py::bytes(f));
    result["frames"] = frames;
    result["packets"] = ps;''')
    s = once(s, 'process_calls > 4096', 'process_calls > 200000')
    s = once(s, 'conf["max_steps"_] = 160;', 'conf["max_steps"_] = 3000;')
    s = once(s, 'conf["max_options"_] = 128;', 'conf["max_options"_] = 256;')
    s = once(s, 'int msg, int places)', 'int msg, int places, int sort_cards, int counters)')
    s = once(s, 'int n = (msg == 18 || msg == 24) ? 3 * places : b[0] + 1;',
             'int n = msg == 25 ? (b[0] == 0xff ? 1 : sort_cards) : msg == 22 ? 2*counters : (msg == 18 || msg == 24) ? 3 * places : b[0] + 1;')
    s = once(s, 'query_field_card(d, p, loc, 3, b.data(), 0)',
             'query_field_card(d, p, loc, 0x00ebc3fb, b.data(), 0)')
    # These fields come from the same staged selection that encodes core responses.
    s = once(s, 'item["place"] = int(a.place_); item["response"] = a.response_;', '''item["place"] = int(a.place_); item["response"] = a.response_;
      item["place_name"] = action_place_to_string(a.place_);
      item["number"] = a.number_; item["attribute"] = a.attribute_; item["race"] = a.race_;''')
    s = once(s, 'result["process_calls"] = process_calls;', '''result["process_calls"] = process_calls;
    result["termination_reason"] = int(termination_reason_);
    result["turn_count"] = turn_count_;
    result["selection"] = py::dict("mode"_a=ms_mode_, "index"_a=ms_idx_,
      "min"_a=ms_min_, "max"_a=ms_max_, "must"_a=ms_must_,
      "selected"_a=ms_r_idxs_, "counter_card"_a=ms_counter_card_,
      "counter_low"_a=ms_counter_low_, "counter_high"_a=ms_counter_high_,
      "counter_remaining"_a=ms_counter_remaining_);''')
    # Avoid literal namespace conflicts with the adapter's own named keys.
    s = s.replace('"mode"_a', 'py::arg("mode")').replace('"index"_a', 'py::arg("index")')
    for name in ('min','max','must','selected','counter_card','counter_low','counter_high','counter_remaining'):
        s = s.replace('"'+name+'"_a', 'py::arg("'+name+'")')
    s = once(s, 'PYBIND11_MODULE(exercise_actor_native, m)', 'PYBIND11_MODULE(jev_duel_native, m)')
    s = once(s, '(m, "ExerciseActor")', '(m, "Duel")')
    (a.output/'jev_duel_bridge.cpp').write_text(s)
    # Reuse previously verified portable objects; no production module replacement.
    build = Path('scripts/exercises/build_actor_bridge.sh').read_text()
    build = 'set -euo pipefail\nout="${JEV_BUILD:?}"\n' + build[build.index('lua='):]
    start = build.index('for file in')
    end = build.index('done\n', start)+5
    build = build[:start]+build[end:]
    build = build.replace('scripts/exercises/actor_bridge.cpp', '"$out/jev_duel_bridge.cpp"')
    build = build.replace('exercise_actor_native.cpython', 'jev_duel_native.cpython')
    (a.output/'build.sh').write_text(build)
    manifest = {str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in (header, a.output/'jev_duel_bridge.cpp', a.output/'build.sh')}
    (a.output/'jev-build-manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
