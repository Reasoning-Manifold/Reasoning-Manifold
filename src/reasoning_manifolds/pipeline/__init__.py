"""Multi-GPU pipeline: launcher, worker, aggregator.

Reproduces the per-prompt extraction + per-layer ID/V + ℋ pipeline from the
paper. Replaces the duplicated logic in ``intelligence/{launcher,worker,
aggregator,main}.py``.

CLI entry: ``reasoning-manifolds run ...`` (see ``launcher.main``).
"""
