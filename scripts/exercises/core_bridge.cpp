// Isolated test bridge. Link the pinned core; never replace the actor extension.
#include "ocgapi.h"
#define API extern "C" __attribute__((visibility("default")))
API void ex_set_script_reader(script_reader f) { set_script_reader(f); }
API void ex_set_card_reader(card_reader f) { set_card_reader(f); }
API void ex_set_message_handler(message_handler f) { set_message_handler(f); }
API intptr_t ex_create_duel(uint_fast32_t s) { return create_duel(s); }
API void ex_end_duel(intptr_t d) { end_duel(d); }
API void ex_start_duel(intptr_t d, int32 o) { start_duel(d, o); }
API int32 ex_preload_script(intptr_t d, const char* s, int32 n) { return preload_script(d, s, n); }
API uint32 ex_process(intptr_t d) { return process(d); }
API int32 ex_get_message(intptr_t d, byte* b) { return get_message(d, b); }
API void ex_get_log_message(intptr_t d, char* b) { get_log_message(d, b); }
API void ex_set_responsei(intptr_t d, int32 v) { set_responsei(d, v); }
API void ex_set_responseb(intptr_t d, byte* b) { set_responseb(d, b); }
API int32 ex_query_field_card(intptr_t d, uint8 p, uint8 l, uint32 f, byte* b, int32 c) {
  return query_field_card(d, p, l, f, b, c);
}
