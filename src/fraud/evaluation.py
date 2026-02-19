"""Ranking metrics that match how a fraud team runs the queue.

A fraud model is judged by what an analyst can act on: how many true frauds sit
in the top N alerts (precision@k), how much recall is available at a fixed
false-positive budget, and whether the score is monotone enough for a threshold
to be meaningful (ROC AUC, average precision).
"""
