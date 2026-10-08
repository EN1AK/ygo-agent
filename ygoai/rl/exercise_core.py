"""Bounded, teacher-only real-core fixture runner (pinned legacy core ABI).

Not the policy observation/action interface. Unsupported messages fail closed.
Callbacks are process-global in this core: run one instance per process.
"""
import ctypes as C
import hashlib
import sqlite3
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CardData(C.Structure):
    _fields_ = [('code', C.c_uint32), ('alias', C.c_uint32),
                ('setcode', C.c_uint16 * 16)] + [
        (k, C.c_uint32) for k in ('type', 'level', 'attribute', 'race')
    ] + [('attack', C.c_int32), ('defense', C.c_int32)] + [
        (k, C.c_uint32) for k in ('lscale', 'rscale', 'link_marker')]


class Reader:
    def __init__(self, data):
        self.data, self.i = data, 0

    def take(self, n):
        b = self.data[self.i:self.i + n]
        if len(b) != n:
            raise ValueError('truncated core message')
        self.i += n
        return b

    def u8(self):
        return self.take(1)[0]

    def u32(self):
        return int.from_bytes(self.take(4), 'little')

    def cards(self, extra=0):
        result = []
        for _ in range(self.u8()):
            row = dict(code=self.u32(), player=self.u8(), location=self.u8(), sequence=self.u8())
            if extra:
                row['extra'] = int.from_bytes(self.take(extra), 'little')
            result.append(row)
        return result


# Payload lengths, not including opcode. Variable layouts are handled below.
FIXED = {1: 0, 2: 6, 5: 2, 32: 1, 35: 1, 37: 0, 38: 6,
         40: 1, 41: 2, 50: 16, 53: 9, 54: 8, 55: 16, 56: 4,
         60: 8, 61: 0, 62: 8, 63: 0, 64: 8, 65: 0,
         70: 16, 71: 1, 72: 1, 73: 1, 74: 0, 75: 1, 76: 1,
         91: 5, 92: 5, 93: 8, 94: 5, 96: 8, 97: 8, 100: 5,
         101: 7, 102: 7, 110: 8, 111: 26, 112: 0, 113: 0, 114: 0,
         120: 8, 160: 9, 165: 6, 170: 4}
PROMPTS = {10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 24, 26}


def messages(data):
    r = Reader(data)
    while r.i < len(data):
        start = r.i
        op = r.u8()
        m = {'op': op}
        if op in (10, 11):
            m['player'] = r.u8()
            if op == 11:
                for name in ('summon', 'special', 'reposition', 'set_monster', 'set_spell', 'activate'):
                    m[name] = r.cards(4 if name == 'activate' else 0)
                m['battle'], m['end'], m['shuffle'] = r.u8(), r.u8(), r.u8()
            else:
                m['activate'], m['attack'] = r.cards(4), r.cards(1)
                m['main2'], m['end'] = r.u8(), r.u8()
        elif op in (12, 13):
            m['player'] = r.u8()
            if op == 12:
                m['code'], m['location'] = r.u32(), r.u32()
            m['description'] = r.u32()
        elif op == 14:
            m['player'] = r.u8()
            m['options'] = [r.u32() for _ in range(r.u8())]
        elif op in (15, 20):
            m['player'], m['cancel'], m['min'], m['max'] = (r.u8() for _ in range(4))
            m['cards'] = r.cards(1)
        elif op == 16:
            m['player'], count, m['special_count'], m['forced'] = (r.u8() for _ in range(4))
            r.take(8)
            m['cards'] = []
            for _ in range(count):
                m['cards'].append(dict(flag=r.u8(), code=r.u32(), player=r.u8(),
                                      location=r.u8(), sequence=r.u8(), position=r.u8(), description=r.u32()))
        elif op in (18, 24):
            m['player'], m['count'], m['mask'] = r.u8(), r.u8(), r.u32()
        elif op == 19:
            m['player'], m['code'], m['positions'] = r.u8(), r.u32(), r.u8()
        elif op == 26:
            m['player'], m['finish'], m['cancel'], m['min'], m['max'] = (r.u8() for _ in range(5))
            m['cards'], m['unselect'] = r.cards(1), r.cards(1)
        elif op in (30, 31, 42):
            m['player'] = r.u8()
            m['cards'] = r.cards()
        elif op in (33, 39, 90):
            r.u8()
            r.take(4 * r.u8())
        elif op == 83:
            r.take(4 * r.u8())
        elif op == 81:
            r.u8()
            r.take(4 * r.u8())
        elif op == 162:
            r.u8()
            for _ in range(2):
                r.take(4)
                for _ in range(7):
                    if r.u8():
                        r.take(2)
                for _ in range(8):
                    if r.u8():
                        r.take(1)
                r.take(6)
            r.take(15 * r.u8())
        elif op in (163, 164):
            r.take(int.from_bytes(r.take(2), 'little') + 1)
        elif op in FIXED:
            r.take(FIXED[op])
        else:
            raise ValueError(f'unsupported core message {op} at {start}: {data.hex()}')
        m['raw'] = data[start:r.i].hex()
        yield m


class Core:
    def __init__(self, library, database, scripts, seed=1):
        self.library = C.CDLL(str(Path(library).resolve()))
        self.scripts = Path(scripts)
        self.assets = {}
        self.buffers = {}
        self.errors = []
        self.events = []
        self.pending = []
        self.decisions = []
        self.phase = 0
        self.turn_player = 0
        self.lp = [8000, 8000]
        self.winner = None
        self.d = None
        self.calls = 0
        db = sqlite3.connect(f'file:{Path(database).resolve().as_posix()}?mode=ro', uri=True)
        db.row_factory = sqlite3.Row
        self.cards = {row['id']: dict(row) for row in db.execute('select * from datas')}
        db.close()
        self.runtime = {'core_sha256': sha(library), 'database_sha256': sha(database), 'seed': seed}
        signatures = {
            'create_duel': (C.c_ssize_t, [C.c_ulong]),
            'end_duel': (None, [C.c_ssize_t]),
            'start_duel': (None, [C.c_ssize_t, C.c_int]),
            'preload_script': (C.c_int, [C.c_ssize_t, C.c_char_p, C.c_int]),
            'process': (C.c_uint32, [C.c_ssize_t]),
            'get_message': (C.c_int, [C.c_ssize_t, C.c_void_p]),
            'get_log_message': (None, [C.c_ssize_t, C.c_void_p]),
            'set_responsei': (None, [C.c_ssize_t, C.c_int]),
            'set_responseb': (None, [C.c_ssize_t, C.c_void_p]),
            'query_field_card': (C.c_int, [C.c_ssize_t, C.c_uint8, C.c_uint8, C.c_uint32, C.c_void_p, C.c_int]),
        }
        for name, (ret, args) in signatures.items():
            f = getattr(self.library, 'ex_' + name)
            f.restype, f.argtypes = ret, args
            setattr(self, name, f)
        self.read_script = C.CFUNCTYPE(C.c_void_p, C.c_char_p, C.POINTER(C.c_int))(self._script)
        self.read_card = C.CFUNCTYPE(C.c_uint32, C.c_uint32, C.POINTER(CardData))(self._card)
        self.log = C.CFUNCTYPE(C.c_uint32, C.c_ssize_t, C.c_uint32)(self._log)
        for name, callback in [('script_reader', self.read_script), ('card_reader', self.read_card), ('message_handler', self.log)]:
            f = getattr(self.library, 'ex_set_' + name)
            f.argtypes = [type(callback)]
            f(callback)
        self.d = self.create_duel(seed)
        self._check()

    def _script(self, name, length):
        try:
            key = name.decode().removeprefix('./script/').removeprefix('script/')
            if key == 'c0.lua':
                length[0] = 0
                return None
            if key not in self.buffers:
                path = (self.scripts / key).resolve()
                if not path.is_relative_to(self.scripts.resolve()):
                    raise ValueError('script path escaped asset directory')
                data = path.read_bytes()
                self.assets[key] = hashlib.sha256(data).hexdigest()
                self.buffers[key] = (C.create_string_buffer(data), len(data))
            buf, n = self.buffers[key]
            length[0] = n
            return C.addressof(buf)
        except Exception as e:
            self.errors.append(str(e))
            length[0] = 0
            return None

    def _card(self, code, ptr):
        try:
            x = CardData()
            x.code = code
            if code in (10000050, 10000060, 10000070):
                x.setcode[0], x.type = 0xa1, 0x4000
            elif code:
                row = self.cards[code]
                for k in ('alias', 'type', 'attribute', 'race'):
                    setattr(x, k, row[k])
                for i in range(4):
                    x.setcode[i] = (row['setcode'] >> (16 * i)) & 65535
                x.level, x.lscale, x.rscale = row['level'] & 255, (row['level'] >> 24) & 255, (row['level'] >> 16) & 255
                x.attack = row['atk']
                if row['type'] & 0x4000000:
                    x.link_marker = row['def']
                else:
                    x.defense = row['def']
            ptr[0] = x
        except Exception as e:
            self.errors.append(f'card {code}: {e}')
        return 0

    def _log(self, duel, kind):
        b = C.create_string_buffer(65536)
        self.get_log_message(duel, b)
        self.errors.append(b.value.decode(errors='replace'))
        return 0

    def _check(self):
        if self.errors:
            raise ValueError('core errors: ' + '; '.join(self.errors))

    def start(self, lua, lp):
        self.lp = list(lp)
        b = lua.encode()
        name = b'__exercise_fixture.lua'
        self.buffers[name.decode()] = (C.create_string_buffer(b), len(b))
        self.assets[name.decode()] = hashlib.sha256(b).hexdigest()
        if not self.preload_script(self.d, name, len(name)):
            self._check()
            raise ValueError('puzzle preload rejected')
        self._check()
        self.start_duel(self.d, 2 | (4 << 16))

    def next_prompt(self, limit=4096):
        while self.winner is None:
            if not self.pending:
                if self.calls >= limit:
                    raise TimeoutError('core process budget exhausted')
                self.calls += 1
                result = self.process(self.d)
                self._check()
                b = C.create_string_buffer(65536)
                n = self.get_message(self.d, b)
                self.pending = list(messages(b.raw[:n]))
                if not self.pending:
                    if result & 0x20000000:
                        raise ValueError('core ended without result')
                    continue
            m = self.pending.pop(0)
            self.events.append(m)
            raw = bytes.fromhex(m['raw'])
            op = m['op']
            if op == 1:
                raise ValueError('engine rejected previous response')
            if op == 5:
                self.winner = raw[1]
            elif op == 40:
                self.turn_player = raw[1]
            elif op == 41:
                self.phase = int.from_bytes(raw[1:3], 'little')
            elif op in (91, 92, 94, 100):
                p, amount = raw[1], int.from_bytes(raw[2:6], 'little')
                self.lp[p] = amount if op == 94 else self.lp[p] + (amount if op == 92 else -amount)
            if op in PROMPTS:
                return m
        return None

    def respond(self, prompt, value):
        self.decisions.append({'prompt': prompt, 'response': value if isinstance(value, int) else value.hex(),
                               'event_index': len(self.events) - 1})
        if isinstance(value, int):
            self.set_responsei(self.d, value)
        else:
            if len(value) > 64:
                raise ValueError('oversized response')
            self.set_responseb(self.d, C.create_string_buffer(value, 64))

    def state(self):
        result = {'lp': self.lp[:], 'phase': self.phase, 'turn_player': self.turn_player,
                  'winner': self.winner, 'cards': []}
        for player in (0, 1):
            for location in (1, 2, 4, 8, 16, 32, 64):
                b = C.create_string_buffer(65536)
                n = self.query_field_card(self.d, player, location, 3, b, 0)
                r = Reader(b.raw[:n])
                while r.i < n:
                    size = r.u32()
                    if size == 4:
                        continue
                    data = Reader(r.take(size - 4))
                    if data.u32() != 3:
                        raise ValueError('unexpected query flags')
                    result['cards'].append(dict(code=data.u32(), player=data.u8(),
                                                location=data.u8(), sequence=data.u8(), position=data.u8()))
        return result

    def close(self):
        if self.d is not None:
            self.end_duel(self.d)
            self.d = None
