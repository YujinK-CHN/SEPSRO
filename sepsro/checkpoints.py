"""
Policy checkpoints that store every network once.

A checkpoint is the joint policy (agent -> Policy) one evaluation scored. Most
of its bytes are torch networks, and most networks recur from one checkpoint to
the next: a PSRO population member stays in every later mixture, a GEMS
generator stays until the next iteration ends. So each network (torch.nn.Module)
is written once, to <dir>/objects/<hash>.pkl, named by the hash of its pickled
contents, and the checkpoint file holds the rest of the policy plus references
to those objects. Identical contents -> one file, however many checkpoints
(or anchors within one checkpoint) use it; a network that changed gets a new
file, so a checkpoint never points at a later state of it.

    save_policies(policies, path, store=True)   # store=False: one plain pickle
    load_policies(path)                         # either kind

Only load files your own runs wrote (pickle).
"""

from __future__ import annotations

import hashlib
import os
import pickle

OBJECTS_DIR = 'objects'


def _module_type():
    try:
        import torch.nn
        return torch.nn.Module
    except ImportError:   # no torch: no networks to share
        return None


def _write_atomic(path, data):
    tmp = f'{path}.tmp{os.getpid()}'
    with open(tmp, 'wb') as handle:
        handle.write(data)
    os.replace(tmp, path)


class _StorePickler(pickle.Pickler):
    """Writes each torch module to the object store and pickles a reference instead."""

    def __init__(self, file, objects_dir):
        super().__init__(file, protocol=pickle.HIGHEST_PROTOCOL)
        self.objects_dir = objects_dir
        self.module_type = _module_type()
        self._keys = {}   # id -> key, for modules referenced more than once in this file

    def persistent_id(self, obj):
        if self.module_type is None or not isinstance(obj, self.module_type):
            return None
        key = self._keys.get(id(obj))
        if key is None:
            data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            key = hashlib.sha256(data).hexdigest()[:24]
            path = os.path.join(self.objects_dir, f'{key}.pkl')
            if not os.path.exists(path):
                _write_atomic(path, data)
            self._keys[id(obj)] = key
        return ('module', key)


class _StoreUnpickler(pickle.Unpickler):

    def __init__(self, file, objects_dir):
        super().__init__(file)
        self.objects_dir = objects_dir
        self._loaded = {}   # one object per key, so shared modules stay shared

    def persistent_load(self, pid):
        kind, key = pid
        if kind != 'module':
            raise pickle.UnpicklingError(f'unknown reference {pid!r}')
        if key not in self._loaded:
            with open(os.path.join(self.objects_dir, f'{key}.pkl'), 'rb') as handle:
                self._loaded[key] = pickle.load(handle)
        return self._loaded[key]


def objects_dir(checkpoint_path):
    return os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), OBJECTS_DIR)


def save_policies(policies, path, store=True):
    """Pickle a joint policy; with store, its networks go to the shared object store."""
    if not store:
        _write_atomic(path, pickle.dumps(policies, protocol=pickle.HIGHEST_PROTOCOL))
        return
    directory = objects_dir(path)
    os.makedirs(directory, exist_ok=True)
    import io
    buffer = io.BytesIO()
    _StorePickler(buffer, directory).dump(policies)
    _write_atomic(path, buffer.getvalue())


def load_policies(path):
    """The joint policy a run saved (with or without the object store)."""
    with open(path, 'rb') as handle:
        return _StoreUnpickler(handle, objects_dir(path)).load()
