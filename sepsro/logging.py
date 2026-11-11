"""
Run logging, in JSONL: one JSON object per line, append-only.

Each record carries its own keys, so heterogeneous record types never end up
under the wrong CSV header, and a file from a killed run still parses up to its
last complete line.

Two streams per run directory:

    eval.jsonl   the plotting surface. One record per evaluation watermark,
                 identical schema for every method, so all methods can be read
                 into one frame and plotted on one budget axis.
    train.jsonl  everything method-internal (losses, meta-strategies,
                 population sizes, ...). Heterogeneous by design; for diagnosis.

Every record of both streams carries `wall_time` (local ISO time) and
`elapsed_s` (seconds since the logger was made), so a run's speed and where it
spent its time can be read from the files. progress() prints a timestamped
line to stdout only (the Slurm log).
"""

from __future__ import annotations

import json
import os
import time

import numpy as np


def _coerce(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


class JsonlWriter:
    """Append-only JSONL sink, flushed per record so a killed run keeps its log."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self._handle = None

    def write(self, record):
        if self._handle is None:
            self._handle = open(self.path, 'a', encoding='utf-8')
        self._handle.write(json.dumps(record, default=_coerce) + '\n')
        self._handle.flush()

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class RunLogger:

    def __init__(self, run_dir, meta=None, echo=True):
        self.run_dir = run_dir
        # Stamped on every eval record. Anything that distinguishes one
        # experimental arm from another belongs here: two runs that differ only
        # by a field missing from meta are indistinguishable to the plotter and
        # get averaged as if they were seeds of one condition.
        self.meta = dict(meta or {})
        self.echo = echo
        self._eval = JsonlWriter(os.path.join(run_dir, 'eval.jsonl'))
        self._train = JsonlWriter(os.path.join(run_dir, 'train.jsonl'))
        self._start = time.time()

    def _stamp(self):
        return {'wall_time': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'elapsed_s': round(time.time() - self._start, 1)}

    def _prefix(self):
        return f'{time.strftime("%m-%d %H:%M:%S")} +{(time.time() - self._start) / 3600:.2f}h'

    def eval(self, record):
        record = {**self.meta, **record, **self._stamp()}
        self._eval.write(record)
        if self.echo:
            shown = {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in record.items() if k not in self.meta and k not in self._stamp()}
            print(f'[eval {self._prefix()}] {shown}', flush=True)

    def train(self, kind, record):
        self._train.write({'kind': kind, **record, **self._stamp()})

    def progress(self, message):
        """A timestamped line in stdout (the Slurm log), not in the JSONL files."""
        if self.echo:
            print(f'[{self._prefix()}] {message}', flush=True)

    def close(self):
        self._eval.close()
        self._train.close()
