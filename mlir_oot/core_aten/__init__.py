"""Core ATen device lowerings for stock Gemmini (direct RoCC only, no loop FSM).

Recipes are written as static primitive streams (``stream.Stream``), checked by
``lint`` against the stock-RTL hazard rules, compressed into ordinary CPU loops
by ``reroll`` and emitted through the existing Gemmini dialect and device
lowering (``emit``).  ``matchers`` binds model2MLIR linalg forms to recipes and
``catalog`` publishes the bound kernels for Merlin.
"""
