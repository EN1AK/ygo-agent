import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from ygoai.windbot_supervisor import classify_result, run_attempt, run_batch


EPISODE = 'Episode 1: length=68, reward=4.0, win=1, win_reason=1\n'
COMPLETE = EPISODE + 'len=68.0000, reward=4.0000, win_rate=1.0000, win_reason=1.0000\n'
STRUCTURED = (
    'Episode 1: length=68, reward=4.0, win=1, win_reason=1, '
    'invalid_game=0, termination_reason=1, episode_steps=68, turn_count=2\n'
    'len=68.0000, reward=4.0000, win_rate=1.0000, win_reason=1.0000, '
    'natural_games=1, invalid_games=0, invalid_rate=0.0000\n'
)


class ResultTests(unittest.TestCase):
    def test_signed_terminal_reason_is_valid(self):
        self.assertEqual(classify_result(0, COMPLETE.replace('win_reason=1\n', 'win_reason=-1\n'))['status'], 'valid')

    def test_success_requires_one_finished_episode(self):
        self.assertEqual(classify_result(0, COMPLETE)['status'], 'valid')
        for text in ['', EPISODE, COMPLETE + EPISODE]:
            self.assertEqual(classify_result(0, text)['status'], 'invalid')

    def test_nonzero_exit_cannot_be_a_win(self):
        result = classify_result(-6, COMPLETE)
        self.assertEqual(result['status'], 'invalid')
        self.assertIsNone(result['episode'])

    def test_nonfinite_reward_cannot_be_valid(self):
        self.assertEqual(classify_result(0, COMPLETE.replace('reward=4.0,', 'reward=nan,'))['status'], 'invalid')

    def test_structured_episode_and_invalid_flag(self):
        self.assertEqual(classify_result(0, STRUCTURED)['status'], 'valid')
        invalid = classify_result(0, STRUCTURED.replace('invalid_game=0,', 'invalid_game=1,'))
        self.assertEqual((invalid['status'], invalid['reason']), ('invalid', 'invalid_episode'))

    def test_current_structured_environment_index_and_reason(self):
        current = STRUCTURED.replace('turn_count=2\n', 'turn_count=2, environment_index=0\n')
        self.assertEqual(classify_result(0, current)['status'], 'valid')
        for bad in (current.replace('environment_index=0', 'environment_index=1'),
                    current.replace('termination_reason=1', 'termination_reason=2'),
                    current.replace('invalid_game=0', 'invalid_game=1')):
            self.assertEqual(classify_result(0, bad)['status'], 'invalid')


@unittest.skipUnless(sys.platform == 'linux', 'Linux process-group integration')
class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def command(self, code):
        return [sys.executable, '-u', '-c', code]

    def test_parallel_requires_isolation(self):
        with self.assertRaises(ValueError):
            run_batch([], self.root / 'unsafe', self.root, 5, workers=2)
        self.assertFalse((self.root / 'unsafe').exists())

    def test_parallel_preserves_slow_attempt_failures_and_replays(self):
        jobs = []
        for i, delay in enumerate((.4, .01, .05, .01)):
            code = ('import time,os; from pathlib import Path; '
                    f'time.sleep({delay}); Path("replay").mkdir(); '
                    'Path("replay/same.yrp").write_bytes(b"fixture"); '
                    + ('os._exit(3)' if i == 2 else 'print(' + repr(COMPLETE) + ')'))
            jobs.append({'command': self.command(code), 'seed': i // 2, 'player': i % 2})
        output = self.root / 'parallel'
        summary = run_batch(jobs, output, self.root, 5, workers=2, isolate_workers=True)
        self.assertEqual((summary['attempted'], summary['valid'], summary['invalid']), (4, 3, 1))
        rows = sorted((json.loads(s) for s in (output / 'results.jsonl').read_text().splitlines()),
                      key=lambda r: r['attempt'])
        self.assertEqual([r['attempt'] for r in rows], [1, 2, 3, 4])
        self.assertEqual([r['metadata']['seed'] for r in rows], [0, 0, 1, 1])
        for i in range(1, 5):
            self.assertTrue((output / f'attempt-{i:04d}/replay/same.yrp').exists())

    def test_parallel_interrupt_cleans_inflight_without_starting_rest(self):
        output = self.root / 'cancel-parallel'
        code = ('from ygoai.windbot_supervisor import run_batch; import sys; '
                'run_batch([{"command":[sys.executable,"-c","import time; time.sleep(60)"],'
                '"seed":i} for i in range(20)],'
                + repr(str(output)) + ',' + repr(str(self.root))
                + ',60, workers=2, isolate_workers=True)')
        p = subprocess.Popen(self.command(code))
        try:
            deadline = time.monotonic() + 10
            while len(list(output.glob('attempt-*/worker.json'))) < 2 and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertEqual(len(list(output.glob('attempt-*/worker.json'))), 2)
            p.send_signal(signal.SIGTERM)
            p.wait(timeout=10)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['attempted'], 2)
            self.assertTrue(summary['interrupted'])
            self.assertTrue(summary['cleanup_complete'])
            rows = [json.loads(s) for s in (output / 'results.jsonl').read_text().splitlines()]
            self.assertEqual(sorted(r['metadata']['seed'] for r in rows), [0, 1])
            self.assertTrue(all(r['reason'] == 'interrupted' for r in rows))
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()

    def test_isolated_workers_preserve_same_named_replays(self):
        code = ('from pathlib import Path; Path("replay").mkdir(); '
                'Path("replay/same.yrp").write_bytes(b"fixture"); print(' + repr(COMPLETE) + ')')
        jobs = [{'command': self.command(code)} for _ in range(2)]
        output = self.root / 'isolated'
        result = run_batch(jobs, output, self.root, 5, isolate_workers=True)
        self.assertEqual(result['valid'], 2)
        for index in (1, 2):
            self.assertEqual((output / f'attempt-{index:04d}/replay/same.yrp').read_bytes(), b'fixture')
        self.assertFalse((self.root / 'replay').exists())

    def test_child_exception_invalidates_completed_episode(self):
        folder = self.root / 'child-error'
        code = ('from pathlib import Path; p=Path(' + repr(str(folder / 'windbot')) + '); '
                'p.mkdir(); (p/"bot.stderr.log").write_text("System.IO.EndOfStreamException: bad chain"); '
                'print(' + repr(COMPLETE) + ')')
        result = run_attempt(self.command(code), folder, self.root, timeout=5)
        self.assertEqual(result['reason'], 'windbot_exception')
        self.assertIsNone(result['episode'])

    def test_failure_then_fresh_success_excludes_invalid_from_win_rate(self):
        jobs = [{'command': self.command('import os; os._exit(3)'), 'seed': 1},
                {'command': self.command('print(' + repr(COMPLETE) + ')'), 'seed': 2}]
        summary = run_batch(jobs, self.root / 'batch', self.root, timeout=5)
        self.assertEqual((summary['attempted'], summary['valid'], summary['invalid']), (2, 1, 1))
        self.assertEqual(summary['win_rate'], 1.0)
        self.assertEqual(summary['wins'], 1)

    def test_timeout_cleans_owned_group(self):
        result = run_attempt(self.command('import time; time.sleep(60)'), self.root / 'timeout', self.root, timeout=.15)
        self.assertEqual(result['reason'], 'timeout')
        self.assertTrue(result['cleanup_complete'])
        self.assertIsNone(result['episode'])

    def test_aborting_evaluator_cleans_stopped_child(self):
        child = 'import os,signal; os.kill(os.getpid(),signal.SIGSTOP)'
        code = ('import os,subprocess,sys,time; from pathlib import Path; '
                'p=subprocess.Popen([sys.executable,"-c",' + repr(child) + ']); '
                'Path("child.pid").write_text(str(p.pid)); time.sleep(.15); os.abort()')
        result = run_attempt(self.command(code), self.root / 'abort', self.root, timeout=5)
        self.assertEqual(result['exit_code'], -signal.SIGABRT)
        self.assertEqual(result['status'], 'invalid')
        self.assertTrue(result['cleanup_complete'])
        pid = int((self.root / 'child.pid').read_text())
        stat = Path(f'/proc/{pid}/stat')
        self.assertTrue(not stat.exists() or stat.read_text().split(') ', 1)[1][0] == 'Z')

    def test_supervisor_termination_records_interrupted_and_cleans_child(self):
        code = ('from ygoai.windbot_supervisor import run_batch; import sys; '
                'run_batch([{"command":[sys.executable,"-c","import time; time.sleep(60)"]}],'
                + repr(str(self.root / 'signal')) + ',' + repr(str(self.root)) + ',60)')
        p = subprocess.Popen(self.command(code))
        try:
            path = self.root / 'signal/attempt-0001/command.json'
            deadline = time.monotonic() + 5
            while not path.exists() and time.monotonic() < deadline:
                time.sleep(.02)
            time.sleep(.2)
            p.send_signal(signal.SIGTERM)
            p.wait(timeout=5)
            result = json.loads((self.root / 'signal/attempt-0001/result.json').read_text())
            self.assertEqual(result['reason'], 'interrupted')
            self.assertTrue(result['cleanup_complete'])
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()

    def test_cli_schedules_seats_seeds_and_keeps_failures_separate(self):
        root = self.root / 'fake-repo'
        (root / 'scripts').mkdir(parents=True)
        (root / 'scripts/eval.py').write_text(
            'import sys,os\n'
            'if sys.argv[sys.argv.index("--player")+1]=="1": os._exit(3)\n'
            'print(' + repr(COMPLETE) + ')\n')
        cli = Path(__file__).resolve().parents[1] / 'scripts/eval_windbot.py'
        output = self.root / 'cli'
        process = subprocess.run([sys.executable, str(cli), '--repo-root', str(root),
                                  '--games', '3', '--alternate-seats', '--seed', '7',
                                  '--output', str(output)], capture_output=True, text=True)
        self.assertEqual(process.returncode, 1, process.stderr)
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual((summary['valid'], summary['invalid']), (1, 2))
        rows = [json.loads(line) for line in (output / 'results.jsonl').read_text().splitlines()]
        self.assertEqual([r['metadata']['seed'] for r in rows], [7, 7, 8])
        self.assertEqual([r['metadata']['player'] for r in rows], [1, 0, 1])
        for i in range(1, 4):
            command = json.loads((output / f'attempt-{i:04d}/command.json').read_text())
            self.assertEqual(command[command.index('--windbot-port') + 1], '0')


if __name__ == '__main__':
    unittest.main()
