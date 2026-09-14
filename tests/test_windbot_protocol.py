import struct
import unittest
import socket

from ygoai.windbot_protocol import LegacyChainProxy, translate_server_packet


class ChainProtocolTests(unittest.TestCase):
    def test_confirm_cards_inserts_panel_flag_preserving_revealed_cards(self):
        cards = struct.pack('<IBBB', 123456, 1, 2, 0)
        legacy = bytes([1, 31, 0, 1]) + cards
        self.assertEqual(translate_server_packet(legacy), bytes([1, 31, 0, 0, 1]) + cards)
        with self.assertRaises(ValueError):
            translate_server_packet(legacy[:-1])

    def test_fragmented_frames_and_response_bytes(self):
        with socket.socket() as host:
            host.bind(('127.0.0.1', 0))
            host.listen(1)
            host.settimeout(3)
            proxy = LegacyChainProxy('127.0.0.1', host.getsockname()[1], 3)
            proxy.start()
            self.addCleanup(proxy.stop)
            with socket.create_connection(('127.0.0.1', proxy.port), 3) as client:
                upstream, _ = host.accept()
                with upstream:
                    upstream.settimeout(3)
                    packet = bytes([1, 16, 0, 0, 0, 0]) + bytes(8)
                    frame = len(packet).to_bytes(2, 'little') + packet
                    for byte in frame:
                        upstream.sendall(bytes([byte]))
                    expected = translate_server_packet(packet)
                    data = b''
                    while len(data) < len(expected) + 2:
                        data += client.recv(100)
                    self.assertEqual(data, len(expected).to_bytes(2, 'little') + expected)
                    response = b'\x05\x00\x01\xff\xff\xff\xff'
                    client.sendall(response)
                    data = b''
                    while len(data) < len(response):
                        data += upstream.recv(100)
                    self.assertEqual(data, response)
            proxy.stop()
            self.assertIsNone(proxy.error)

    def test_forced_flag_moves_to_each_candidate_without_reordering(self):
        for forced in (0, 1):
            entries = [struct.pack('<BIBBBBI', 2, 123456, 0, 4, 3, 1, 777),
                       struct.pack('<BIBBBBI', 0, 987654, 1, 8, 4, 2, 888)]
            header = bytes([1, 16, 1, 2, 0])
            hints = struct.pack('<II', 0x11223344, 0x55667788)
            old = header + bytes([forced]) + hints + b''.join(entries)
            expected = header + hints + b''.join(e[:1] + bytes([forced]) + e[1:] for e in entries)
            self.assertEqual(translate_server_packet(old), expected)

    def test_empty_chain_and_unrelated_message(self):
        self.assertEqual(translate_server_packet(bytes([1, 16, 0, 0, 0, 0]) + bytes(8)),
                         bytes([1, 16, 0, 0, 0]) + bytes(8))
        self.assertEqual(translate_server_packet(b'\x01\x05hello'), b'\x01\x05hello')

    def test_truncated_chain_is_rejected(self):
        with self.assertRaises(ValueError):
            translate_server_packet(bytes([1, 16, 0, 1, 0, 1]) + bytes(8))
