"""Legacy YGOPro global chain force flag -> WindBot per-candidate flags.

SELECT_CHAIN and CONFIRM_CARDS are adapted to the deployed WindBot reader.
Client responses and other messages are forwarded verbatim.
"""
import selectors
import socket
import threading


def translate_server_packet(packet):
    if packet[:2] == b'\x01\x1f':
        if len(packet) < 4 or len(packet) != 4 + packet[3] * 7:
            raise ValueError('Unexpected legacy CONFIRM_CARDS length')
        # New reader has a UI-only skip_panel byte after player. Preserve every
        # revealed card and its location, with the normal panel behavior (0).
        return packet[:3] + b'\x00' + packet[3:]
    if packet[:2] != b'\x01\x10':
        return packet
    if len(packet) < 14:
        raise ValueError('Truncated legacy SELECT_CHAIN header')
    count, forced = packet[3], packet[5]
    if len(packet) != 14 + count * 13:
        raise ValueError('Unexpected legacy SELECT_CHAIN candidate length')
    entries = packet[14:]
    return packet[:5] + packet[6:14] + b''.join(
        entries[i:i+1] + bytes([forced]) + entries[i+1:i+13]
        for i in range(0, len(entries), 13))


class LegacyChainProxy:
    def __init__(self, host, upstream_port, timeout=30):
        self.host, self.upstream_port, self.timeout = host, upstream_port, timeout
        self.listener = socket.socket()
        self.listener.bind((host, 0))
        self.listener.listen(1)
        self.listener.settimeout(timeout)
        self.port = self.listener.getsockname()[1]
        self.sockets = [self.listener]
        self.error = None
        self.converted = 0
        self.stopping = False
        self.thread = threading.Thread(target=self._run, daemon=True, name='windbot-chain-proxy')

    def start(self):
        self.thread.start()

    def stop(self):
        self.stopping = True
        for sock in self.sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        if self.thread.is_alive():
            self.thread.join(timeout=2)

    def _run(self):
        try:
            client, _ = self.listener.accept()
            self.sockets.append(client)
            server = socket.create_connection((self.host, self.upstream_port), self.timeout)
            self.sockets.append(server)
            with selectors.DefaultSelector() as selector:
                selector.register(client, selectors.EVENT_READ, server)
                selector.register(server, selectors.EVENT_READ, client)
                buffers = {client: bytearray(), server: bytearray()}
                while not self.stopping:
                    for key, _ in selector.select(timeout=0.5):
                        data = key.fileobj.recv(65536)
                        if not data:
                            return
                        buf = buffers[key.fileobj]
                        buf.extend(data)
                        while len(buf) >= 2:
                            size = int.from_bytes(buf[:2], 'little')
                            if size < 1:
                                raise ValueError('Empty YGOPro packet')
                            if len(buf) < size + 2:
                                break
                            packet = bytes(buf[2:size+2])
                            del buf[:size+2]
                            if key.fileobj is server:
                                converted = translate_server_packet(packet)
                                self.converted += converted != packet
                                packet = converted
                            key.data.sendall(len(packet).to_bytes(2, 'little') + packet)
        except Exception as exc:
            if not self.stopping:
                self.error = repr(exc)
                print(f'WindBot protocol proxy error: {exc}', flush=True)
        finally:
            for sock in self.sockets:
                sock.close()
