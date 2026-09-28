import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_ygocore_protocol.py"
SPEC = importlib.util.spec_from_file_location("audit_ygocore_protocol", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ProtocolAuditTest(unittest.TestCase):
    def test_source_hash_is_checkout_line_ending_invariant(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "adapter.h"
            source.write_bytes(b"first\r\nsecond\rthird\n")
            mixed_hash = MODULE.sha256_source_file(source)
            source.write_bytes(b"first\nsecond\nthird\n")
            self.assertEqual(MODULE.sha256_source_file(source), mixed_hash)

    def test_message_inventory_rejects_duplicate_values(self):
        with tempfile.TemporaryDirectory() as directory:
            common = Path(directory) / "common.h"
            common.write_text("#define MSG_ONE 1\n#define MSG_TWO 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate MSG value"):
                MODULE.parse_messages(common)

    def test_adapter_inventory_is_limited_to_msg_to_string(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "adapter.h"
            adapter.write_text(
                "case MSG_OUTSIDE: break;\n"
                "static std::string msg_to_string(int msg) {\n"
                "case MSG_INSIDE: return \"inside\";\n"
                "}\n// system string\n",
                encoding="utf-8",
            )
            self.assertEqual(MODULE.adapter_message_names(adapter), {"MSG_INSIDE"})

    def test_adapter_handler_inventory_requires_real_parser_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "adapter.h"
            adapter.write_text(
                "case MSG_NAME_ONLY: break;\n"
                "void handle_message() {\n"
                "  if (msg_ == MSG_ONE) {}\n"
                "  else if (msg_ == MSG_TWO || msg_ == MSG_THREE) {}\n"
                "}\n"
                "void _damage(int player, int amount) {}\n",
                encoding="utf-8",
            )
            self.assertEqual(
                MODULE.adapter_handler_message_names(adapter),
                {"MSG_ONE", "MSG_TWO", "MSG_THREE"},
            )

    def test_source_payload_writes_are_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture.cpp"
            source.write_text(
                "int field::select_x(int step) {\n"
                "  pduel->write_buffer8(MSG_SELECT_CARD);\n"
                "  pduel->write_buffer8(player);\n"
                "  pduel->write_buffer32(code);\n"
                "  return FALSE;\n"
                "}\n",
                encoding="utf-8",
            )
            writers = MODULE.scan_writers(root)
            record = writers["MSG_SELECT_CARD"][0]
            self.assertEqual(record["function"], "field::select_x")
            self.assertEqual(
                record["payload_fields"],
                [
                    {"width_bits": 8, "expression": "player", "source_line": 3},
                    {"width_bits": 32, "expression": "code", "source_line": 4},
                ],
            )

    def test_unknown_message_requires_explicit_classification(self):
        with self.assertRaisesRegex(ValueError, "unclassified core message"):
            MODULE.classify("MSG_FUTURE_PROTOCOL_VALUE")

    def test_production_handler_never_skips_the_remaining_core_buffer(self):
        adapter = Path(__file__).resolve().parents[1] / "ygoenv" / "ygoenv" / "ygopro" / "ygopro.h"
        text = adapter.read_text(encoding="utf-8")
        start = text.index("void handle_message()")
        end = text.index("void _damage(", start)
        self.assertNotIn("dp_ = dl_", text[start:end])


if __name__ == "__main__":
    unittest.main()
