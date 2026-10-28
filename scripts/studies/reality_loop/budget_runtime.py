"""Existing model inputs/actions with strict process-budget prefix adjudication."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'crystron_curriculum'))
from supervision_runtime import Runtime as OriginalRuntime
from engine_budget_audit import eligible, adjudicate


class Runtime(OriginalRuntime):
    def independent(self, record):
        if eligible(record):
            revised = adjudicate(record, self.source, self.args.database, self.args.scripts,
                self.plan['independent_library'], self.plan['native_sha256'])
            record.clear()
            record.update(revised)
            return record['independent_audit']
        return super().independent(record)
