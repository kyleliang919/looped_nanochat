# Validation performed

Pinned upstream: `karpathy/nanochat@92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`.

Checks run in the artifact environment:

- `py_compile` on all four modified Python files and the new test file: passed.
- Patch dry-run and application against the pinned source files: passed.
- Routing-disabled model compared with upstream using an identical state dict/input: exact equality (`max_abs_error = 0`).
- Augmented-Q/K SDPA compared with an explicit additive routing-bias reference: `max_abs_error = 3.5762787e-07`.
- Tiny routing model forward/backward: routing gates and pass-1 source Q projections received finite nonzero gradients after nonzero output-projection test initialization.
- Cached full prefill compared with uncached full forward: exact equality in the tiny test.
- Incremental final-token decode compared with uncached full forward: `max_abs_error <= 3.73e-09` in the tiny test.
- Optimizer parameter grouping checked with routing disabled and enabled: passed.

The complete upstream nanochat test suite was not run because the execution environment could not clone the full repository. The included `tests/test_routing_loop.py` is intended to be run after applying the patch to a normal nanochat checkout.
