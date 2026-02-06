"""Cost-matrix threshold selection.

A 0.5 probability threshold silently assumes a false positive and a missed
fraud cost the same. They do not: reviewing a flagged transaction costs minutes
of an analyst's time, while a missed fraud costs the transaction amount plus
chargeback handling. This module turns that trade-off into one number, the
expected loss per transaction, and picks the threshold that minimises it.
"""
