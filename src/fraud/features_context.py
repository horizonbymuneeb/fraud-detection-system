"""Context features: time of day, merchant category risk, and first sightings.

Time is encoded cyclically (sin/cos) rather than as a raw integer so 23:00 and
00:00 sit next to each other, and a raw day-of-week column so weekly patterns
stay interpretable. Merchant-category risk is a fixed domain table, overridable
from ``config.yaml``; category frequency is fitted on the training window only.
"""
