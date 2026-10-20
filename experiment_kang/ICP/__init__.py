"""ICP-inspired unpaired cross-modal alignment experiments.

A: Robust ICP (trimmed, confidence-weighted pseudo-pairs)
B: Multi-Scale ICP (coarse-to-fine cluster + instance refinement)
C: Covariance-Aware ICP (local uncertainty weighted Procrustes)
D: Structure-Aware ICP (matched-neighborhood graph consistency)
E: Piecewise Orthogonal ICP (locally adaptive orthogonal maps)
F: Hybrid ICP (A + B + D)
"""

METHODS = ("original", "A", "B", "C", "D", "E", "F")
