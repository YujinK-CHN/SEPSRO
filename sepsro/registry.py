"""
Name -> class registries.

A method is selected by --method and an evaluation metric by --eval-metrics.
Registering by decorator means adding a new one is a new file plus one import
in its package __init__, never an edit to main.py.

    from sepsro.registry import METHODS

    @METHODS.register('psro')
    class PSRO(Method):
        ...
"""


class Registry:

    def __init__(self, kind):
        self.kind = kind
        self._entries = {}

    def register(self, name):
        def decorator(obj):
            if name in self._entries:
                raise ValueError(f'{self.kind} {name!r} is registered twice.')
            self._entries[name] = obj
            return obj
        return decorator

    def get(self, name):
        if name not in self._entries:
            raise ValueError(f'Unknown {self.kind} {name!r}. Choose one of {self.names()}.')
        return self._entries[name]

    def names(self):
        return sorted(self._entries)

    def __contains__(self, name):
        return name in self._entries


METHODS = Registry('method')
METRICS = Registry('metric')
