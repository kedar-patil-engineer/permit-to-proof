"""Offline evaluation harness for Permit-to-Proof.

Computes the four metrics the research paper needs (Part A5) against a gold
answer key: extraction precision/recall/F1, the true verification lift (errors
caught with the deterministic layer ON vs OFF), confidence calibration
(ECE + reliability diagram), and the selective-prediction automation-vs-human
trade-off curve. Pure functions live in metrics.py; rendering in report.py.

This package is for the paper's evaluation. The app runs without it.
"""
