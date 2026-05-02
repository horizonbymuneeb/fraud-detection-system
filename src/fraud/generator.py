"""Synthetic card-transaction generator with labelled fraud patterns.

The generator exists so the pipeline can be exercised end to end without a
data licence. It is deliberately not a toy: cards have a home country, a
preferred category mix, a spend scale and a personal activity rate, fraud is
injected at a configurable base rate, and each injected ring belongs to a named
pattern so recall can be reported per pattern instead of as one number.
"""
