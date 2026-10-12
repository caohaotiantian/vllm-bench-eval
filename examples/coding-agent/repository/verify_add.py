"""Verify the example task without external dependencies."""
from add import add

assert add(2, 3) == 5
assert add(-2, 3) == 1
