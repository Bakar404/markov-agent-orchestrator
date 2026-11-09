# verified/

Three safety-critical rules of the Markov Agent Orchestrator, stated as requirements, specified
in [Verus](https://github.com/verus-lang/verus), and implemented in Rust that is **proved**
against that specification. A differential test then checks that the **Python** the arena runs
does exactly what the proved Rust does.

The method is Proof-Driven Development (Microsoft Research and GitHub Next's PDD Kit):

1. **Requirements first.** Each rule is written down in plain language (the table below) before
   any proof.
2. **The spec states them.** Each requirement becomes one or more `ensures` clauses in
   `src/lib.rs`, tagged with its id (`// G3`).
3. **The implementation is proved** against the spec by Verus and Z3.
4. **Where the proof can't reach, tests and review cover it.** The Python isn't proved; a
   Hypothesis differential test ties it to the proved kernel. The trust boundaries below say
   exactly what is taken on trust.

```
python verified/check.py        # prove -> mutants -> build -> differential -> control -> receipt
```

## The rules

| Id | Requirement | Python source | Covered by |
|----|-------------|---------------|------------|
| **Gate** | | | |
| G1 | Before escalation, no specialist and no coalition is legal: the legal set is within {invoke_generalist, escalate, terminate}. | `engine.py` `legal_actions` | proved; differential |
| G2 | Before escalation, invoke_generalist is always legal. | same | proved; differential |
| G3 | escalate is legal exactly when the run hasn't escalated, `allow_escalation` is set, and `solo_steps >= min_solo_steps`. | same | proved; differential |
| G4 | After escalation, invoke_generalist and escalate are never legal. | same | proved; differential |
| G5 | After escalation, all six single specialists are legal. | same | proved; differential |
| G6 | run_parallel is legal exactly when escalated with more than 12% of the budget and more than 10% of the latency left. | same | proved; differential |
| G7 | terminate is legal exactly when `step >= min_steps_before_terminate` and the work is done (`unresolved_subtasks == 0`) or the budget or latency is under 25%. | `engine.py` `_may_terminate` | proved; differential |
| **Budget** | | | |
| B1 | What's left (`budget_remaining_usd`) is never negative. | `state.py` | proved; differential |
| B2 | It never exceeds the total, while the total and the spend are non-negative. | same | proved; differential |
| B3 | It is exactly `max(total - spent, 0)`. | same | proved; differential |
| B4 | The normalized remainder (`budget_remaining`, `latency_remaining`) is in [0, 1]: [0, 1 000 000] micro-units. | same | proved; differential |
| B5 | Below the total it is `remaining / total` rounded down to a micro-unit; at or above it, 1; with no positive total, 1 if anything is left, else 0. | same | proved; differential (to float error, TB2) |
| B6 | Spending is monotone: a cost is added exactly (a negative one clamped to 0), saturating at the largest representable amount. | `transitions.py`, `live.py` | proved; differential |
| B7 | So spending never raises what's left, or its normalized form. | same | proved |
| **Guard** | | | |
| R1 | The patch has at least one edit. | `wsp_guard.py` | proved (W1, W2); differential |
| R2 | Every edit names a known tensor: `q_weights`, `synergy`, `terminate_weights`. | same | proved; differential |
| R3 | Every index has one coordinate per dimension (rank), each an int in bounds. | same | proved; differential |
| R4 | A synergy edit is strictly above the diagonal, `i < j`. The policy reads only that triangle, so an edit on or below it changes nothing yet counts against R9. | same | proved; differential |
| R5 | Every value is finite. | same | proved; differential |
| R6 | `|value| <= max_abs_value`. | same | proved; differential |
| R7 | The entry being replaced is finite and `|value - current| <= max_abs_delta`. | same | proved; differential |
| R8 | No two edits target the same `(tensor, index)`. | same | proved; differential |
| R9 | `len(edits) <= floor(max_changed_fraction x total entries of the three tensors)`. The denominator is every stored entry, including the lower triangle and the generalist's synergy row, which the policy never reads; `wsp/apply.py` reports its headline changed-fraction over the 151 effective parameters instead, but the limit is the proved one. | same | proved; differential |
| W1 | An accepted patch satisfies every rule, R1 to R9. | `check_patch` | proved |
| W2 | A rejected patch is rejected with the id of a rule it breaks, and the first one in R1..R9 order: every earlier rule holds. | `check_patch` | proved; differential (rule id) |
| W3 | For R2 to R8 the verdict names the first offending edit: it breaks the rule and every earlier edit keeps it. For R8 it names the first repeat: it shares an earlier edit's target, and no two edits before it share one. | `first_violation` | proved (both halves); differential (edit index) |
| W4 | Applying an accepted patch preserves every tensor's shape. | `apply_patch` | proved; differential |
| W5 | Only the entries the patch targets change, so at most the R9 bound of entries change. | `apply_patch` | proved |
| W6 | Every edit lands: its entry holds its value afterwards. | `apply_patch` | proved; differential |

Rules are checked in order R1 to R9 and the first failure wins. Within a rule the first
offending edit is reported. "Rule-major" is a choice that W2 pins down: a patch whose edit 0
breaks R7 and whose edit 1 breaks R3 is rejected under R3.

### What the gate allows, derived from the code

`engine.py` builds the legal set like this:

| | always | when |
|---|---|---|
| **before escalation** | invoke_generalist | **escalate** if `allow_escalation and solo_steps >= min_solo_steps`; **terminate** if G7 |
| **after escalation** | the six specialists: planner, researcher, critic, verifier, executor, memory | **run_parallel** if more than 12% of the budget and more than 10% of the latency are left; **terminate** if G7 |

Both thresholds are decided on micro-unit amounts, exactly as the kernel decides them:
`state.compare_remaining` quantises the total and the spend, then compares
`remaining x 1e6` against `threshold x total` in integers.

The brief asks for "before escalation the legal set is *exactly* {invoke_generalist, escalate,
terminate}". The code doesn't do that: escalate and terminate are conditional (G3, G7), so the
set is a *subset* of those three, and it always contains invoke_generalist. The proof states
what the code does (G1 to G3 and G7). Requiring the literal "exactly" would mean making
escalate and terminate unconditional, which would remove the `min_solo_steps` and
`_may_terminate` guards. That is a behaviour change, so it is not made here.

## What is proved

`src/lib.rs`, inside `verus! { }`, is proved by Verus 0.2026.11.02.8bf68ee (66 verified, 0 errors
when this was written; `target/receipt.json` records each run). That covers:

* `legal_actions(&GateInput) -> Vec<bool>`: the mask over `ACTIONS`, in `actions.py` order.
  The thresholds are compared exactly over rationals, with no floats.
* `remaining`, `normalized` and `spend` over i64 micro-units, plus a lemma that the normalized
  remainder is monotone in the spend.
* `check_patch(&Shapes, &Vec<i64>, &Limits, &Vec<Edit>) -> Verdict`, with one exec check per
  rule, each proved equal to its spec predicate. Also `apply_patch`, proved under `requires
  holds_through(9)`, over the flat `[q_weights | synergy | terminate_weights]` row-major layout.
  The layout is proved injective, so distinct targets never collide.
* Freedom from overflow and panics in all of the above. Every arithmetic operation and index is
  checked by Verus.

## What is tested, not proved

The Python is not proved. `backend/tests/test_verified_kernel.py` runs the proved kernel as a
CLI (`src/main.rs`, JSON lines) and asserts **Python == Rust** on Hypothesis-generated inputs,
plus 37 named edge cases. Inputs are biased toward the boundaries:

* **Gate:** budgets and latencies sitting exactly on the 12%, 10% and 25% thresholds, ±2 micro,
  on the micro grid and as arbitrary floats off it (a float a hair past a threshold, such as
  1.0 total with 0.8799996 spent). Also exhausted resources, zero and negative totals, and both
  escalation states. Python must equal Rust everywhere, ties included.
* **Budget:** remaining and normalized for budget and latency. Spending through both the
  escalation path and the live-report path; negative reports are rejected, while the kernel's
  defensive clamp is checked independently.
* **Guard:** unknown tensor names and non-strings, wrong arity, out-of-bounds, negative, bool,
  float and 2**70 indices, the synergy diagonal and lower triangle, and NaN, ±inf and 1e300
  values. Non-numeric values, values exactly at the limits, half-micro offsets, duplicates,
  non-finite current entries, and R9 bounds of 0. numpy integer and past-i64 indices too.
* **Production apply:** `wsp/apply.py` `apply_patch` (what run creation calls) against the
  real policy's shapes. It must reject under the kernel's rule, or produce the kernel's tensors.
* **Quantization:** every float, including the double-rounding witnesses above 2**52.

A coverage test fails unless every rule R1 to R9 and acceptance were all reached, along with
gate ties, off-grid gate inputs, and both outcomes of the production apply. At the
default 300 examples per property every rule is hit several times; set `VERIFIED_EXAMPLES`
higher to search harder.

## The mutation check

A proof that survives a planted bug is not checking what it claims to. `check.py` plants 29
bugs in a copy of `lib.rs`, at least one per clause; it fails if any tagged clause has none. Each must be **rejected by Verus, on the
clause it breaks**: the error must point at a specification line carrying that clause's tag.
A mutant that doesn't compile counts as a failure, because it tested nothing.

| Clause | Planted bug |
|---|---|
| G1 | specialists are legal before escalation |
| G2 | invoke_generalist depends on `allow_escalation` |
| G3 | escalate ignores `allow_escalation` |
| G4 | a run can escalate twice |
| G5 | invoke_memory is never legal |
| G6 | run_parallel before escalation |
| G7 | terminate ignores `min_steps_before_terminate` |
| B1 / B2 / B3 | remaining not floored / exceeds the total / halved |
| B4 / B5 | normalized exceeds 1 / is not the exact ratio |
| B6 / B7 | a negative cost refunds / a spend that would saturate wraps to zero instead |
| R2 to R9 | one per rule: unknown tensor code accepted, any arity accepted, the diagonal accepted, non-finite accepted, `<=` turned into `<`, delta off by one micro, duplicates missed, bound + 1 |
| W1 / W2 / W3 | one-edit patches accepted unchecked / R9 reported as R8 / the wrong edit named, and the second offending edit named instead of the first |
| W4 / W5 / W6 | apply grows the layout / also zeroes entry 0 / skips the last edit |

## The negative control

A differential test that passes a wrong Python is not checking anything. `check.py` runs the
differential test against four deliberately wrong copies of the Python guard
(`VERIFIED_PY_MUTANT`). Each must make it fail:

| Id | Wrong guard |
|---|---|
| PM1 | R4 lets the synergy diagonal through (`<=` for `<`) |
| PM2 | quantize uses `floor(a + 0.5)`, which double-rounds above 2**52 |
| PM3 | R9 allows one edit too many |
| PM4 | R7 compares raw floats instead of micro-units |

## Trust boundaries

What has to be right for "proved" and "tested" to mean what they say. Nothing outside this list
is assumed.

**TB1. Verus, Z3 and rustc.** Verus and Z3 are trusted to be sound. rustc 1.99.1 is trusted to
compile the proved code to what Verus checked. Both are pinned: Verus by version and SHA256 in CI,
rustc by `rust-toolchain.toml`. The vstd and verus_builtin crates are pinned in `Cargo.toml`,
locked by `Cargo.lock`, and built `--locked`.

**TB2. Quantization.** The kernel reasons about i64 micro-units. The Python uses floats.
`quantize(x)` = `round(x * 1e6)`, rounded half away from zero, saturating at ±4e18, `None` for
NaN or ±inf. It is the same function on both sides (`wsp_guard.quantize`, and `quantize` in
`lib.rs` outside `verus!`). It is trusted rather than proved, because Verus doesn't reason about
f64 arithmetic. The differential test checks the two agree on every float.
Consequences, all by design:

* **The guard rounds values first and compares them second**, in both languages. For example,
  10.0000004 passes `max_abs_value = 10` and 10.0000005 doesn't. `x * 1e6` is itself a float
  product, so "the micro-unit of x" means the rounding of that product.
* **The gate quantises like the kernel.** `engine.py` decides run_parallel and terminate through
  `state.compare_remaining`, which rounds the total and the spend to micro-units with this same
  `quantize` and compares in integers. So Python and Rust agree on every input, exact ties and
  off-grid floats included. (Before, the Python divided floats: with 1.0 total and 0.8799996
  spent it allowed run_parallel, which the kernel, seeing 120 000 of 1 000 000 left, rejects.)
* **The budget figures stay in floats.** `budget_remaining_usd` and the normalized remainders
  are unchanged, so B1 to B7 are differentially tested on the micro grid only.
* Amounts beyond ±4e12 units saturate in the kernel. Guard values that large fail R6 long
  before saturation matters.

**TB3. The Python ↔ Rust adapter.** `src/main.rs` and the `guard_request` / `bits` adapter in
the test are trusted. They map JSON to the proved functions' arguments. Floats travel as IEEE
bits, so nothing is lost on the wire. Non-integer index coordinates (bools included) become -1. Any integer
type the Python guard accepts, numpy's included, travels as a JSON integer; one past i64 is
clamped to the i64 range by the test adapter, which keeps it out of bounds (R3, as in Python)
instead of failing the kernel's JSON parse. Unknown or non-string tensor
names become code 3.

The proved functions have `requires` (finite gate inputs, synergy square, dims ≤ 2**20, ...).
`main.rs` checks each one at runtime and returns an error instead of calling the function.
`wsp_guard.py` validates the same preconditions and raises `ValueError`. Both are reviewed and
tested, not proved.

Resource inputs are checked at the Python boundary: `LiveAgentReport` in `schemas.py` and
`report_from_response` in `live.py` reject non-finite, negative, boolean or over-cap measured
amounts (`agents.py`: cost at most 1e8 in the run's unit, latency at most 1e9 ms, tokens an
integer at most 1e12). `transitions.py` also checks that a step's totals are finite before it
changes any state.
The validation-error handler in `main.py` makes rejected non-finite inputs JSON-safe so the
HTTP response is a clear 422 rather than a serialization failure.
`RunConfig` in `engine.py`, `TransitionModel` in `transitions.py`, and `AgentSpec` in
`agents.py` validate finite, non-negative budgets, escalation costs/latencies, and agent base
resource amounts. These checks establish the input assumptions behind monotone spending;
the validation itself is tested, not proved.

**TB4. Intent → clauses.** Whether G1 to W6 say what the orchestrator *should* do is human
judgment. A proof shows the code meets the clauses, not that the clauses are right. The
requirement table is the thing to review, and the "exactly" gap above is an example of where
intent and code part ways.

## Findings

While writing the spec, four things came up. F-1 to F-3 are now fixed at the Python boundary (F-1 fully closed, upper bounds included);
F-4 remains a wording finding.

* **F-1 (bug, fixed in `backend/app/schemas.py` and `backend/app/orchestration/live.py`):
  an infinite measured latency disabled the latency limit.** Previously `LiveAgentReport.latency_ms`
  was `ge=0` with no finiteness check, and pydantic's JSON parser accepted a bare `Infinity`. The coalition
  formula in `transitions.py` then computes `inf + 0.12 * (inf - inf) = NaN`, so
  `latency_consumed_ms` becomes NaN. Every later comparison against it is false: the episode's
  latency stop (`engine.py`, `latency_consumed_ms >= latency_budget_ms`) never fires, and
  `latency_remaining` is NaN, so the gate's 25% latency trigger for terminate never fires
  either, and run_parallel is never legal again. Pinned by passing rejection regressions in
  `backend/tests/test_verified_findings.py`. Original repro (now raises `ValueError`):

  ```python
  s = initial_state(task_complexity=0.5, budget_usd=1.0, latency_budget_ms=1e4, belief_dim=4,
                    rng=np.random.default_rng(0))
  r = report_from_response(AGENTS["generalist"], s, {"outcome": "partial", "confidence": 0.5,
        "response": "r", "tokens": 1, "cost_usd": 0.0, "latency_ms": float("inf")}, belief_dim=4)
  TransitionModel(belief_dim=4).sample(s, Action.INVOKE_GENERALIST, np.random.default_rng(1),
        agent_ids=["generalist"], reports_override=[r]).next_state.latency_consumed_ms  # nan
  ```

  The fix rejects non-finite and negative latency, cost and token measurements in the schema
  and direct conversion, establishing the finite inputs the spec's `requires` already assumes.
  Finite-but-huge amounts were the same bug by another route: a cost of 1e308 made the reward
  `-inf`, and two latencies of 1e308 summed to `inf`, after which the step committed and every
  response failed to encode. Measurements are now capped (above), tokens must be integers
  within SQLite's range, and `transitions._check_totals` rejects a step whose totals overflow
  before any state moves. `OverflowError` (an int like 10**400) is a 422/400, never a 500.
  `backend/app/main.py` ensures rejected NaN/Infinity inputs can be returned in 422 diagnostics.
* **F-2 (quirk, fixed in `backend/app/orchestration/live.py`): measured zeros were replaced
  by spec defaults.** Only `None` now selects a default; truthful zero latency, tokens and cost
  are preserved, including through the live API.
* **F-3 (gap, fixed in `backend/app/orchestration/engine.py`, `transitions.py`, and `agents.py`):
  escalation and base resource amounts lacked non-negative validation.** Construction now
  rejects non-finite or negative amounts, so B6's monotonicity assumptions hold at this boundary.
* **F-4 (wording): "exactly {G, E, T}" before escalation.** See "What the gate allows" above.

## Files

| Path | What |
|---|---|
| `src/lib.rs` | spec, implementation and proofs (and the trusted `quantize`, outside `verus!`) |
| `src/main.rs` | the `verified-kernel` CLI: JSON lines in, JSON lines out (TB3) |
| `check.py` | the end-to-end check, which writes `target/receipt.json` |
| `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml` | pins |
| `../backend/app/orchestration/wsp_guard.py` | the Python guard, which callers use |
| `../backend/tests/test_verified_kernel.py` | the differential test |
| `../backend/tests/test_verified_findings.py` | passing resource-validation regressions for F-1 and F-3 |

## Running it

You need rustup, the 1.99.1 toolchain, Verus 0.2026.11.02.8bf68ee, and the backend's Python
environment with the dev group (`pip install -r backend/requirements.txt --group
backend/pyproject.toml:dev`, pip ≥ 25.1).

```sh
rustup toolchain install 1.99.1 --profile minimal
export VERUS=/path/to/verus            # or put verus on PATH
python verified/check.py               # full check; exits non-zero on any failure

# pieces
cd verified && "$VERUS" src/lib.rs --crate-type=lib          # prove only
cd verified && cargo build --release --locked                # build the CLI
cd backend && python -m pytest tests/test_verified_kernel.py # differential (skips without cargo)
```

`REQUIRE_VERIFIED=1` turns the differential test's skips (no cargo, no Hypothesis, build failed)
into failures. CI sets it, so the `verified` job can't pass by skipping. `VERIFIED_KERNEL_BIN`
points the test at a prebuilt CLI, and `VERIFIED_EXAMPLES` sets examples per property
(default 300).

The receipt (`target/receipt.json`, uploaded by CI as `verified-receipt`) records:

* the Verus, rustc, Python and Hypothesis versions
* the SHA256 of every checked file
* the verified and error counts
* each mutant and the clauses that rejected it
* the differential test counts, and which rules it exercised
* the negative-control results
* timings

**When the Python changes**, update the clause first, then the proof, then the test. If the
mutation check reports that a mutant's text no longer appears in `lib.rs`, move the mutant to
the new code. Don't delete it.
