"""Typed, validated access to ``config.yaml``.

The rest of the package never reads YAML directly: it receives a :class:`Config`
instance. Keeping parsing and validation in one place means a typo in the config
fails at load time with a message naming the offending key, not deep inside a
feature function.
"""
