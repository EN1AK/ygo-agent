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


if __name__ == "__main__":
    unittest.main()
