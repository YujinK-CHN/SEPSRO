"""
Importing this package registers every method in sepsro.registry.METHODS.

To add a method: create <name>.py here with a class decorated
@METHODS.register('<name>'), import it below, and add its argument group to
config.py.
"""

from . import gems, ppo, psro, random_baseline  # noqa: F401  (registration side effect)
from .base import Method, NotImplementedMethod

__all__ = ['Method', 'NotImplementedMethod']
