//! Three safety rules of the Markov Agent Orchestrator, proved with Verus.
//!
//! 1. The escalation gate: `OrchestrationEngine.legal_actions` in
//!    backend/app/orchestration/engine.py. Clauses G1 to G7.
//! 2. Budget accounting: `budget_remaining_usd`, `budget_remaining` and
//!    `latency_remaining` in backend/app/orchestration/state.py, and the spend in
//!    transitions.py and live.py. Clauses B1 to B7.
//! 3. The WSP weight-patch guard: `check_patch` in
//!    backend/app/orchestration/wsp_guard.py. Clauses R2 to R9 (one per rule) and
//!    W1 to W6 (the verdict, and what applying an accepted patch can do).
//!
//! Each `ensures` clause states one requirement from verified/README.md and ends
//! with its tag, which the mutation check in verified/check.py reads to tell
//! which requirement caught a planted bug. Everything inside `verus!` is proved.
//! The quantizer at the bottom, and the JSON shim in src/main.rs, are outside it
//! and trusted (TB2 and TB3 in the README); the differential test checks both.
//!
//! Real numbers are fixed-point: a value `x` is the i64 `round(x * 1e6)`,
//! rounded half away from zero (micro-dollars, micro-milliseconds, micro-units
//! of a weight or of a fraction).

// A plain cargo build erases the ghost code, which leaves parameters used only
// in `requires` and the `global` layout assertion looking unused to rustc.
#![allow(unused_variables, unused_braces)]

use vstd::prelude::*;
#[allow(unused_imports)]
use vstd::arithmetic::div_mod::*;

verus! {

global size_of usize == 8;

// ═══ Fixed-point vocabulary ═════════════════════════════════════════════════

/// One whole unit (a dollar, a millisecond, a weight, all of a fraction).
pub const MICRO: i64 = 1_000_000;

/// The largest magnitude a quantized value takes; the quantizer saturates here.
/// Small enough that the difference of two in-range values still fits an i64.
pub const MICRO_MAX: i64 = 4_000_000_000_000_000_000;

/// A quantized NaN or infinity. No in-range value equals it.
pub const NONFINITE: i64 = i64::MIN;

pub open spec fn in_range(x: int) -> bool {
    -MICRO_MAX <= x <= MICRO_MAX
}

pub open spec fn clamp0(x: int) -> int {
    if x < 0 { 0 } else { x }
}

pub open spec fn abs(x: int) -> int {
    if x < 0 { -x } else { x }
}

// ═══ 2. Budget accounting ═══════════════════════════════════════════════════
//
// Money and latency are the same arithmetic: a total, an amount used, and
// what's left. state.py: budget_remaining_usd = max(total - spent, 0), and
// budget_remaining = clip(remaining / max(total, 1e-9), 0, 1). Latency likewise.

/// What's left of a total once `spent` is used: Python's max(total - spent, 0).
pub open spec fn remaining_spec(total: int, spent: int) -> int {
    clamp0(total - spent)
}

/// The fraction left, in micro-units, rounded down. A total of zero or less
/// reads as 1 when anything is left (Python divides by 1e-9 and clips), else 0.
pub open spec fn normalized_spec(total: int, spent: int) -> int {
    let rem = remaining_spec(total, spent);
    if total <= 0 {
        if rem > 0 { MICRO as int } else { 0 }
    } else if rem >= total {
        MICRO as int
    } else {
        (rem * MICRO) / total
    }
}

/// What's left of the budget (or the latency allowance).
pub fn remaining(total: i64, spent: i64) -> (r: i64)
    requires
        in_range(total as int),
        in_range(spent as int),
    ensures
        // B1. What's left is never negative.
        r >= 0, // B1
        // B2. It never exceeds the total, while the total and the spend are non-negative.
        total >= 0 && spent >= 0 ==> r <= total, // B2
        // B3. It is the total less the spend, floored at zero: max(total - spent, 0).
        r as int == remaining_spec(total as int, spent as int), // B3
{
    let d = total - spent;
    if d < 0 { 0 } else { d }
}

/// The normalized remainder, in micro-units of 1.
pub fn normalized(total: i64, spent: i64) -> (n: i64)
    requires
        in_range(total as int),
        in_range(spent as int),
    ensures
        // B4. The normalized remainder is a fraction: between 0 and 1 (1,000,000 micro-units).
        0 <= n <= MICRO, // B4
        // B5. Below the total, it is the exact ratio remaining / total, rounded down.
        total > 0 && remaining_spec(total as int, spent as int) < total ==> n * total <= remaining_spec(total as int, spent as int) * MICRO < (n + 1) * total, // B5
        // B5. At or above the total it is 1; with no positive total, 1 if anything is left, else 0.
        n as int == normalized_spec(total as int, spent as int), // B5
{
    let rem = remaining(total, spent);
    if total <= 0 {
        if rem > 0 { MICRO } else { 0 }
    } else if rem >= total {
        MICRO
    } else {
        proof {
            assert(0 <= (rem as int) * (MICRO as int) <= 8_000_000_000_000_000_000 * 1_000_000) by (nonlinear_arith)
                requires 0 <= rem <= 8_000_000_000_000_000_000int;
        }
        let x: u128 = (rem as u128) * (MICRO as u128);
        let q: u128 = x / (total as u128);
        proof {
            lemma_fundamental_div_mod(x as int, total as int);
            lemma_mod_pos_bound(x as int, total as int);
            lemma_div_is_ordered(x as int, (total as int) * (MICRO as int), total as int);
            lemma_div_multiples_vanish(MICRO as int, total as int);
            assert((rem as int) * (MICRO as int) <= (total as int) * (MICRO as int)) by (nonlinear_arith)
                requires 0 <= rem < total;
            let xi = x as int;
            let qi = q as int;
            let ti = total as int;
            assert(qi * ti <= xi < (qi + 1) * ti) by (nonlinear_arith)
                requires xi == ti * (xi / ti) + xi % ti, 0 <= xi % ti < ti, qi == xi / ti;
        }
        q as i64
    }
}

/// Spending more never leaves more: the normalized remainder can't go up.
proof fn lemma_normalized_monotone(total: int, before: int, after: int)
    requires
        before <= after,
    ensures
        normalized_spec(total, after) <= normalized_spec(total, before),
{
    let ra = remaining_spec(total, before);
    let rb = remaining_spec(total, after);
    assert(rb <= ra);
    if total > 0 {
        assert(0 <= rb * MICRO <= ra * MICRO) by (nonlinear_arith)
            requires 0 <= rb <= ra;
        lemma_div_multiples_vanish(MICRO as int, total);
        if rb < total {
            assert(rb * MICRO <= total * MICRO) by (nonlinear_arith)
                requires 0 <= rb < total;
            lemma_div_is_ordered(rb * MICRO, total * MICRO, total);
            if ra < total {
                lemma_div_is_ordered(rb * MICRO, ra * MICRO, total);
            }
        }
    }
}

/// Record a spend (or a latency) against what's been used so far.
pub fn spend(spent: i64, cost: i64) -> (s: i64)
    requires
        in_range(spent as int),
        in_range(cost as int),
    ensures
        in_range(s as int),
        // B6. Spending is monotone: a negative cost is clamped to zero (live.py clamps it), so the amount spent never goes down.
        s >= spent, // B6
        // B6. Otherwise it adds the cost exactly, saturating at the largest representable amount.
        s as int == (if spent + clamp0(cost as int) > MICRO_MAX { MICRO_MAX as int } else { spent + clamp0(cost as int) }), // B6
        // B7. So what's left, and its normalized form, never go up.
        forall|total: int| #[trigger] remaining_spec(total, s as int) <= remaining_spec(total, spent as int), // B7
        forall|total: int| #[trigger] normalized_spec(total, s as int) <= normalized_spec(total, spent as int), // B7
{
    let c = if cost < 0 { 0 } else { cost };
    let s = if spent > MICRO_MAX - c { MICRO_MAX } else { spent + c };
    proof {
        assert forall|total: int| #[trigger] normalized_spec(total, s as int) <= normalized_spec(total, spent as int) by {
            lemma_normalized_monotone(total, spent as int, s as int);
        }
    }
    s
}

// ═══ 1. The escalation gate ═════════════════════════════════════════════════
//
// engine.py legal_actions. The result is a mask over ACTIONS, in actions.py's
// order: entry a is true exactly when action a is legal.

pub const GENERALIST: usize = 0;
pub const PLANNER: usize = 1;
pub const RESEARCHER: usize = 2;
pub const CRITIC: usize = 3;
pub const VERIFIER: usize = 4;
pub const EXECUTOR: usize = 5;
pub const MEMORY: usize = 6;
pub const RUN_PARALLEL: usize = 7;
pub const ESCALATE: usize = 8;
pub const TERMINATE: usize = 9;
pub const ACTION_COUNT: usize = 10;

/// What legal_actions reads: the state, and the run's config. Budget and
/// latency in micro-units.
pub struct GateInput {
    pub has_escalated: bool,
    pub allow_escalation: bool,
    pub solo_steps: i64,
    pub min_solo_steps: i64,
    pub step: i64,
    pub min_steps_before_terminate: i64,
    pub unresolved_subtasks: i64,
    pub budget_total: i64,
    pub budget_spent: i64,
    pub latency_budget: i64,
    pub latency_consumed: i64,
}

pub open spec fn gate_wf(g: GateInput) -> bool {
    in_range(g.budget_total as int) && in_range(g.budget_spent as int) && in_range(g.latency_budget as int)
        && in_range(g.latency_consumed as int)
}

/// The exact normalized remainder is above `t` micro-units. Compared without
/// rounding, as Python compares its float.
pub open spec fn above(total: int, spent: int, t: int) -> bool {
    let rem = remaining_spec(total, spent);
    if total > 0 { rem * MICRO > t * total } else { (if rem > 0 { MICRO as int } else { 0 }) > t }
}

/// The exact normalized remainder is below `t` micro-units.
pub open spec fn below(total: int, spent: int, t: int) -> bool {
    let rem = remaining_spec(total, spent);
    if total > 0 { rem * MICRO < t * total } else { (if rem > 0 { MICRO as int } else { 0 }) < t }
}

/// engine.py _may_terminate: the work is done, or a resource is under 25%.
pub open spec fn may_terminate(g: GateInput) -> bool {
    g.unresolved_subtasks == 0 || below(g.budget_total as int, g.budget_spent as int, 250_000) || below(
        g.latency_budget as int,
        g.latency_consumed as int,
        250_000,
    )
}

pub open spec fn terminate_legal(g: GateInput) -> bool {
    g.step >= g.min_steps_before_terminate && may_terminate(g)
}

pub open spec fn escalate_legal(g: GateInput) -> bool {
    !g.has_escalated && g.allow_escalation && g.solo_steps >= g.min_solo_steps
}

/// More than 12% of the budget and 10% of the latency allowance remain.
pub open spec fn parallel_affordable(g: GateInput) -> bool {
    above(g.budget_total as int, g.budget_spent as int, 120_000) && above(
        g.latency_budget as int,
        g.latency_consumed as int,
        100_000,
    )
}

fn above_exec(total: i64, spent: i64, t: i64) -> (b: bool)
    requires
        in_range(total as int),
        in_range(spent as int),
        0 <= t <= MICRO,
    ensures
        b == above(total as int, spent as int, t as int),
{
    let rem = remaining(total, spent);
    if total > 0 {
        proof {
            assert(0 <= (rem as int) * (MICRO as int) <= 8_000_000_000_000_000_000 * 1_000_000) by (nonlinear_arith)
                requires 0 <= rem <= 8_000_000_000_000_000_000int;
            assert(0 <= (t as int) * (total as int) <= 1_000_000 * 4_000_000_000_000_000_000) by (nonlinear_arith)
                requires 0 <= t <= 1_000_000, 0 < total <= 4_000_000_000_000_000_000int;
        }
        (rem as i128) * (MICRO as i128) > (t as i128) * (total as i128)
    } else {
        (if rem > 0 { MICRO } else { 0 }) > t
    }
}

fn below_exec(total: i64, spent: i64, t: i64) -> (b: bool)
    requires
        in_range(total as int),
        in_range(spent as int),
        0 <= t <= MICRO,
    ensures
        b == below(total as int, spent as int, t as int),
{
    let rem = remaining(total, spent);
    if total > 0 {
        proof {
            assert(0 <= (rem as int) * (MICRO as int) <= 8_000_000_000_000_000_000 * 1_000_000) by (nonlinear_arith)
                requires 0 <= rem <= 8_000_000_000_000_000_000int;
            assert(0 <= (t as int) * (total as int) <= 1_000_000 * 4_000_000_000_000_000_000) by (nonlinear_arith)
                requires 0 <= t <= 1_000_000, 0 < total <= 4_000_000_000_000_000_000int;
        }
        (rem as i128) * (MICRO as i128) < (t as i128) * (total as i128)
    } else {
        (if rem > 0 { MICRO } else { 0 }) < t
    }
}

/// The legal actions, as a mask over ACTIONS.
pub fn legal_actions(g: &GateInput) -> (r: Vec<bool>)
    requires
        gate_wf(*g),
    ensures
        r@.len() == ACTION_COUNT,
        // G1. Before escalation the generalist works alone: no specialist and no coalition is legal, only invoke_generalist, escalate and terminate.
        !g.has_escalated ==> forall|a: int| GENERALIST < a < ESCALATE ==> !#[trigger] r@[a], // G1
        // G2. Before escalation the generalist is always legal.
        !g.has_escalated ==> r@[GENERALIST as int], // G2
        // G3. Escalate is legal exactly when the run hasn't escalated, allows escalation, and the generalist has had min_solo_steps.
        r@[ESCALATE as int] == escalate_legal(*g), // G3
        // G4. After escalation the generalist is retired, and escalation can't happen again.
        g.has_escalated ==> !r@[GENERALIST as int] && !r@[ESCALATE as int], // G4
        // G5. After escalation every single specialist is legal.
        g.has_escalated ==> forall|a: int| GENERALIST < a < RUN_PARALLEL ==> #[trigger] r@[a], // G5
        // G6. run_parallel is legal exactly when escalated with more than 12% of the budget and 10% of the latency left.
        r@[RUN_PARALLEL as int] == (g.has_escalated && parallel_affordable(*g)), // G6
        // G7. terminate is legal exactly when step >= min_steps_before_terminate and the work is done or a resource is under 25%.
        r@[TERMINATE as int] == terminate_legal(*g), // G7
{
    let escalated = g.has_escalated;
    let specialists = escalated;
    let parallel = escalated && above_exec(g.budget_total, g.budget_spent, 120_000) && above_exec(
        g.latency_budget,
        g.latency_consumed,
        100_000,
    );
    let escalate = !escalated && g.allow_escalation && g.solo_steps >= g.min_solo_steps;
    let terminate = g.step >= g.min_steps_before_terminate && (g.unresolved_subtasks == 0 || below_exec(
        g.budget_total,
        g.budget_spent,
        250_000,
    ) || below_exec(g.latency_budget, g.latency_consumed, 250_000));

    let mut r: Vec<bool> = Vec::new();
    r.push(!escalated);  // invoke_generalist
    r.push(specialists);  // invoke_planner
    r.push(specialists);  // invoke_researcher
    r.push(specialists);  // invoke_critic
    r.push(specialists);  // invoke_verifier
    r.push(specialists);  // invoke_executor
    r.push(specialists);  // invoke_memory
    r.push(parallel);  // run_parallel
    r.push(escalate);  // escalate
    r.push(terminate);  // terminate
    r
}

// ═══ 3. The WSP weight-patch guard ══════════════════════════════════════════
//
// wsp_guard.py check_patch. The three tensors the markov_game policy reads sit
// in one flat row-major layout: q_weights (rows x cols), then synergy (n x n),
// then terminate_weights (len).

pub const Q_WEIGHTS: u8 = 0;
pub const SYNERGY: u8 = 1;
pub const TERMINATE_WEIGHTS: u8 = 2;

/// No dimension is larger than this (the Python guard refuses such shapes).
pub const MAX_DIM: u64 = 1_048_576;

pub struct Shapes {
    pub q_rows: u64,
    pub q_cols: u64,
    pub s_n: u64,
    pub t_len: u64,
}

/// One edit. `tensor` is Q_WEIGHTS, SYNERGY, TERMINATE_WEIGHTS, or anything
/// else for a name the policy doesn't have. `value` is quantized, or NONFINITE.
pub struct Edit {
    pub tensor: u8,
    pub index: Vec<i64>,
    pub value: i64,
}

/// GuardLimits, quantized: max_abs_value, max_abs_delta, max_changed_fraction.
pub struct Limits {
    pub max_abs: i64,
    pub max_delta: i64,
    pub max_fraction: i64,
}

/// The verdict. `rule` is 0 for accepted, else the rule number 1 to 9. `edit`
/// is the offending edit for R2 to R8.
pub struct Verdict {
    pub rule: u8,
    pub edit: usize,
}

pub open spec fn q_len(s: Shapes) -> int {
    s.q_rows * s.q_cols
}

pub open spec fn s_len(s: Shapes) -> int {
    s.s_n * s.s_n
}

pub open spec fn total_entries(s: Shapes) -> int {
    q_len(s) + s_len(s) + s.t_len
}

pub open spec fn shapes_wf(s: Shapes) -> bool {
    s.q_rows <= MAX_DIM && s.q_cols <= MAX_DIM && s.s_n <= MAX_DIM && s.t_len <= MAX_DIM
}

/// The guard's inputs are well formed: the current weights fill the layout,
/// every number is in range or NONFINITE, and the limits are valid.
pub open spec fn guard_wf(s: Shapes, cur: Seq<i64>, lim: Limits, edits: Seq<Edit>) -> bool {
    &&& shapes_wf(s)
    &&& cur.len() == total_entries(s)
    &&& forall|p: int| 0 <= p < cur.len() ==> #[trigger] cur[p] == NONFINITE || in_range(cur[p] as int)
    &&& forall|k: int| 0 <= k < edits.len() ==> #[trigger] edits[k].value == NONFINITE || in_range(edits[k].value as int)
    &&& 0 <= lim.max_abs <= MICRO_MAX
    &&& 0 <= lim.max_delta <= MICRO_MAX
    &&& 0 <= lim.max_fraction <= MICRO
}

pub open spec fn known(t: u8) -> bool {
    t == Q_WEIGHTS || t == SYNERGY || t == TERMINATE_WEIGHTS
}

/// The tensor's shape.
pub open spec fn dims(s: Shapes, t: u8) -> Seq<int> {
    if t == Q_WEIGHTS {
        seq![s.q_rows as int, s.q_cols as int]
    } else if t == SYNERGY {
        seq![s.s_n as int, s.s_n as int]
    } else {
        seq![s.t_len as int]
    }
}

/// Where an in-bounds edit lands in the flat layout.
pub open spec fn pos(s: Shapes, e: Edit) -> int {
    if e.tensor == Q_WEIGHTS {
        e.index@[0] * s.q_cols + e.index@[1]
    } else if e.tensor == SYNERGY {
        q_len(s) + e.index@[0] * s.s_n + e.index@[1]
    } else {
        q_len(s) + s_len(s) + e.index@[0]
    }
}

/// R2. The edit names a tensor the policy has: q_weights, synergy or terminate_weights.
pub open spec fn rule2(e: Edit) -> bool {
    known(e.tensor)
}

/// R3. The index has one coordinate per dimension, and each is in bounds.
pub open spec fn rule3(s: Shapes, e: Edit) -> bool {
    &&& known(e.tensor)
    &&& e.index@.len() == dims(s, e.tensor).len()
    &&& forall|d: int| 0 <= d < e.index@.len() ==> 0 <= #[trigger] e.index@[d] < dims(s, e.tensor)[d]
}

/// R4. A synergy edit is strictly above the diagonal (i < j): the policy reads
/// only the upper triangle.
pub open spec fn rule4(e: Edit) -> bool {
    e.tensor == SYNERGY ==> e.index@[0] < e.index@[1]
}

/// R5. The value is finite.
pub open spec fn rule5(e: Edit) -> bool {
    e.value != NONFINITE
}

/// R6. |value| <= max_abs_value.
pub open spec fn rule6(lim: Limits, e: Edit) -> bool {
    abs(e.value as int) <= lim.max_abs
}

/// R7. The entry it replaces is finite, and |value - current| <= max_abs_delta.
pub open spec fn rule7(s: Shapes, cur: Seq<i64>, lim: Limits, e: Edit) -> bool {
    &&& cur[pos(s, e)] != NONFINITE
    &&& abs(e.value - cur[pos(s, e)]) <= lim.max_delta
}

pub open spec fn same_target(a: Edit, b: Edit) -> bool {
    a.tensor == b.tensor && a.index@ == b.index@
}

/// R8. No two edits target the same (tensor, index).
pub open spec fn rule8(edits: Seq<Edit>) -> bool {
    forall|k1: int, k2: int| 0 <= k1 < k2 < edits.len() ==> !same_target(#[trigger] edits[k1], #[trigger] edits[k2])
}

/// R9's bound: floor(max_changed_fraction x total entries).
pub open spec fn change_bound(s: Shapes, lim: Limits) -> int {
    (lim.max_fraction * total_entries(s)) / (MICRO as int)
}

/// R9. At most change_bound edits.
pub open spec fn rule9(s: Shapes, lim: Limits, edits: Seq<Edit>) -> bool {
    edits.len() <= change_bound(s, lim)
}

/// The per-edit rules, R2 to R7, by number.
pub open spec fn edit_passes(r: int, s: Shapes, cur: Seq<i64>, lim: Limits, e: Edit) -> bool {
    if r == 2 {
        rule2(e)
    } else if r == 3 {
        rule3(s, e)
    } else if r == 4 {
        rule4(e)
    } else if r == 5 {
        rule5(e)
    } else if r == 6 {
        rule6(lim, e)
    } else if r == 7 {
        rule7(s, cur, lim, e)
    } else {
        true
    }
}

/// Rule r holds for the whole patch. R1: the patch is not empty.
pub open spec fn rule_holds(r: int, s: Shapes, cur: Seq<i64>, lim: Limits, edits: Seq<Edit>) -> bool {
    if r == 1 {
        edits.len() > 0
    } else if 2 <= r <= 7 {
        forall|k: int| 0 <= k < edits.len() ==> edit_passes(r, s, cur, lim, #[trigger] edits[k])
    } else if r == 8 {
        rule8(edits)
    } else if r == 9 {
        rule9(s, lim, edits)
    } else {
        true
    }
}

/// Rules R1 to Rn all hold.
pub open spec fn holds_through(n: int, s: Shapes, cur: Seq<i64>, lim: Limits, edits: Seq<Edit>) -> bool {
    forall|r: int| 1 <= r <= n ==> #[trigger] rule_holds(r, s, cur, lim, edits)
}

/// The entries a patch targets, one per edit.
pub open spec fn targets(s: Shapes, edits: Seq<Edit>) -> Seq<int> {
    Seq::new(edits.len(), |k: int| pos(s, edits[k]))
}

/// The entries that differ between two layouts.
pub open spec fn changed(before: Seq<i64>, after: Seq<i64>) -> Set<int> {
    vstd::set_lib::set_int_range(0, before.len() as int).filter(|p: int| before[p] != after[p])
}

// ─── Rule checks, one per rule ──────────────────────────────────────────────

fn check_known(e: &Edit) -> (ok: bool)
    ensures
        ok == rule2(*e), // R2
{
    e.tensor == Q_WEIGHTS || e.tensor == SYNERGY || e.tensor == TERMINATE_WEIGHTS
}

fn in_bounds(i: i64, dim: u64) -> (ok: bool)
    ensures
        ok == (0 <= i < dim),
{
    i >= 0 && (i as u64) < dim
}

fn check_index(s: &Shapes, e: &Edit) -> (ok: bool)
    ensures
        ok == rule3(*s, *e), // R3
{
    let ix = &e.index;
    let ok = if e.tensor == Q_WEIGHTS {
        ix.len() == 2 && in_bounds(ix[0], s.q_rows) && in_bounds(ix[1], s.q_cols)
    } else if e.tensor == SYNERGY {
        ix.len() == 2 && in_bounds(ix[0], s.s_n) && in_bounds(ix[1], s.s_n)
    } else if e.tensor == TERMINATE_WEIGHTS {
        ix.len() == 1 && in_bounds(ix[0], s.t_len)
    } else {
        false
    };
    proof {
        if ok {
            assert forall|d: int| 0 <= d < e.index@.len() implies 0 <= #[trigger] e.index@[d] < dims(*s, e.tensor)[d] by {
                if e.tensor != TERMINATE_WEIGHTS {
                    assert(d == 0 || d == 1);
                }
            }
        } else if known(e.tensor) && e.index@.len() == dims(*s, e.tensor).len() {
            if e.tensor == TERMINATE_WEIGHTS {
                assert(!(0 <= e.index@[0] < dims(*s, e.tensor)[0]));
            } else if !(0 <= e.index@[0] < dims(*s, e.tensor)[0]) {
            } else {
                assert(!(0 <= e.index@[1] < dims(*s, e.tensor)[1]));
            }
        }
    }
    ok
}

fn check_upper(s: &Shapes, e: &Edit) -> (ok: bool)
    requires
        rule3(*s, *e),
    ensures
        ok == rule4(*e), // R4
{
    e.tensor != SYNERGY || e.index[0] < e.index[1]
}

fn check_finite(e: &Edit) -> (ok: bool)
    ensures
        ok == rule5(*e), // R5
{
    e.value != NONFINITE
}

fn check_magnitude(lim: &Limits, e: &Edit) -> (ok: bool)
    requires
        rule5(*e),
        in_range(e.value as int),
    ensures
        ok == rule6(*lim, *e), // R6
{
    let a = if e.value < 0 { -e.value } else { e.value };
    a <= lim.max_abs
}

/// The flat position of an in-bounds edit.
fn position(s: &Shapes, e: &Edit) -> (p: usize)
    requires
        shapes_wf(*s),
        rule3(*s, *e),
    ensures
        p as int == pos(*s, *e),
        0 <= pos(*s, *e) < total_entries(*s),
{
    proof {
        lemma_total_bound(*s);
        lemma_pos_bounds(*s, *e);
        assert(0 <= e.index@[0] < dims(*s, e.tensor)[0]);
        if e.tensor != TERMINATE_WEIGHTS {
            assert(0 <= e.index@[1] < dims(*s, e.tensor)[1]);
        }
        if e.tensor == SYNERGY {
            assert(0 <= e.index@[0] * s.s_n <= e.index@[0] * s.s_n + e.index@[1]) by (nonlinear_arith)
                requires 0 <= e.index@[0], s.s_n >= 0, 0 <= e.index@[1];
        } else if e.tensor == Q_WEIGHTS {
            assert(0 <= e.index@[0] * s.q_cols <= e.index@[0] * s.q_cols + e.index@[1]) by (nonlinear_arith)
                requires 0 <= e.index@[0], s.q_cols >= 0, 0 <= e.index@[1];
        }
    }
    let ql = s.q_rows * s.q_cols;
    let sl = s.s_n * s.s_n;
    if e.tensor == Q_WEIGHTS {
        (e.index[0] as u64 * s.q_cols + e.index[1] as u64) as usize
    } else if e.tensor == SYNERGY {
        (ql + e.index[0] as u64 * s.s_n + e.index[1] as u64) as usize
    } else {
        (ql + sl + e.index[0] as u64) as usize
    }
}

fn check_delta(s: &Shapes, cur: &Vec<i64>, lim: &Limits, e: &Edit) -> (ok: bool)
    requires
        guard_wf(*s, cur@, *lim, seq![]),
        rule3(*s, *e),
        rule5(*e),
        in_range(e.value as int),
    ensures
        ok == rule7(*s, cur@, *lim, *e), // R7
{
    let c = cur[position(s, e)];
    if c == NONFINITE {
        false
    } else {
        let d = e.value - c;
        let a = if d < 0 { -d } else { d };
        a <= lim.max_delta
    }
}

/// Dispatch to the check for rule r.
fn edit_passes_exec(r: u8, s: &Shapes, cur: &Vec<i64>, lim: &Limits, e: &Edit) -> (ok: bool)
    requires
        2 <= r <= 7,
        guard_wf(*s, cur@, *lim, seq![]),
        e.value == NONFINITE || in_range(e.value as int),
        r >= 4 ==> rule3(*s, *e),
        r >= 6 ==> rule5(*e),
    ensures
        ok == edit_passes(r as int, *s, cur@, *lim, *e),
{
    if r == 2 {
        check_known(e)
    } else if r == 3 {
        check_index(s, e)
    } else if r == 4 {
        check_upper(s, e)
    } else if r == 5 {
        check_finite(e)
    } else if r == 6 {
        check_magnitude(lim, e)
    } else {
        check_delta(s, cur, lim, e)
    }
}

/// The first edit that breaks rule r, if any.
fn first_failing(r: u8, s: &Shapes, cur: &Vec<i64>, lim: &Limits, edits: &Vec<Edit>) -> (res: Option<usize>)
    requires
        2 <= r <= 7,
        guard_wf(*s, cur@, *lim, edits@),
        holds_through(r - 1, *s, cur@, *lim, edits@),
    ensures
        match res {
            None => rule_holds(r as int, *s, cur@, *lim, edits@),
            Some(k) => k < edits@.len() && !edit_passes(r as int, *s, cur@, *lim, edits@[k as int]),
        },
        // W3. The edit returned is the first to break rule r: every earlier edit passes it.
        res matches Some(k) ==> forall|j: int| 0 <= j < k ==> edit_passes(r as int, *s, cur@, *lim, #[trigger] edits@[j]), // W3
{
    let mut k: usize = 0;
    while k < edits.len()
        invariant
            2 <= r <= 7,
            k <= edits@.len(),
            guard_wf(*s, cur@, *lim, edits@),
            holds_through(r - 1, *s, cur@, *lim, edits@),
            forall|j: int| 0 <= j < k ==> edit_passes(r as int, *s, cur@, *lim, #[trigger] edits@[j]),
        decreases edits@.len() - k,
    {
        proof {
            let e = edits@[k as int];
            assert(e.value == NONFINITE || in_range(e.value as int));
            if r >= 4 {
                assert(rule_holds(3, *s, cur@, *lim, edits@));
                assert(edit_passes(3, *s, cur@, *lim, e));
            }
            if r >= 6 {
                assert(rule_holds(5, *s, cur@, *lim, edits@));
                assert(edit_passes(5, *s, cur@, *lim, e));
            }
        }
        if !edit_passes_exec(r, s, cur, lim, &edits[k]) {
            return Some(k);
        }
        k = k + 1;
    }
    None
}

fn same_index(a: &Vec<i64>, b: &Vec<i64>) -> (eq: bool)
    ensures
        eq == (a@ == b@),
{
    if a.len() != b.len() {
        return false;
    }
    let mut i: usize = 0;
    while i < a.len()
        invariant
            a@.len() == b@.len(),
            i <= a@.len(),
            forall|j: int| 0 <= j < i ==> a@[j] == b@[j],
        decreases a@.len() - i,
    {
        if a[i] != b[i] {
            return false;
        }
        i = i + 1;
    }
    proof {
        assert(a@ =~= b@);
    }
    true
}

/// R8: the first edit that repeats an earlier edit's target, if any.
fn first_duplicate(edits: &Vec<Edit>) -> (res: Option<usize>)
    ensures
        match res {
            None => rule8(edits@),
            Some(k2) => k2 < edits@.len() && exists|k1: int| 0 <= k1 < k2 && same_target(#[trigger] edits@[k1], edits@[k2 as int]),
        }, // R8
        // W3. The edit returned is the first repeat: no two edits before it share a target.
        res matches Some(k2) ==> forall|a: int, b: int| 0 <= a < b < k2 ==> !same_target(#[trigger] edits@[a], #[trigger] edits@[b]), // W3
{
    let n = edits.len();
    let mut k2: usize = 0;
    while k2 < n
        invariant
            n == edits@.len(),
            k2 <= n,
            forall|a: int, b: int| 0 <= a < b < k2 ==> !same_target(#[trigger] edits@[a], #[trigger] edits@[b]),
        decreases n - k2,
    {
        let mut k1: usize = 0;
        while k1 < k2
            invariant
                n == edits@.len(),
                k1 <= k2 < n,
                forall|a: int, b: int| 0 <= a < b < k2 ==> !same_target(#[trigger] edits@[a], #[trigger] edits@[b]),
                forall|a: int| 0 <= a < k1 ==> !same_target(#[trigger] edits@[a], edits@[k2 as int]),
            decreases k2 - k1,
        {
            if edits[k1].tensor == edits[k2].tensor && same_index(&edits[k1].index, &edits[k2].index) {
                return Some(k2);
            }
            k1 = k1 + 1;
        }
        proof {
            assert forall|a: int, b: int| 0 <= a < b < k2 + 1 implies !same_target(#[trigger] edits@[a], #[trigger] edits@[b]) by {
                if b == k2 {
                    assert(!same_target(edits@[a], edits@[k2 as int]));
                }
            }
        }
        k2 = k2 + 1;
    }
    None
}

/// R9: floor(max_changed_fraction x total entries).
fn change_bound_exec(s: &Shapes, lim: &Limits) -> (b: u128)
    requires
        shapes_wf(*s),
        0 <= lim.max_fraction <= MICRO,
    ensures
        b as int == change_bound(*s, *lim), // R9
{
    proof {
        lemma_total_bound(*s);
        assert(0 <= (lim.max_fraction as int) * total_entries(*s) <= 1_000_000 * 3_000_000_000_000) by (nonlinear_arith)
            requires 0 <= lim.max_fraction <= 1_000_000, 0 <= total_entries(*s) <= 3_000_000_000_000int;
    }
    let total: u128 = (s.q_rows as u128) * (s.q_cols as u128) + (s.s_n as u128) * (s.s_n as u128) + (s.t_len as u128);
    (lim.max_fraction as u128) * total / (MICRO as u128)
}

proof fn lemma_total_bound(s: Shapes)
    requires
        shapes_wf(s),
    ensures
        0 <= q_len(s) <= 1_099_511_627_776,
        0 <= s_len(s) <= 1_099_511_627_776,
        0 <= total_entries(s) <= 3_000_000_000_000,
{
    assert(0 <= s.q_rows * s.q_cols <= 1_048_576 * 1_048_576) by (nonlinear_arith)
        requires s.q_rows <= 1_048_576, s.q_cols <= 1_048_576, s.q_rows >= 0, s.q_cols >= 0;
    assert(0 <= s.s_n * s.s_n <= 1_048_576 * 1_048_576) by (nonlinear_arith)
        requires s.s_n <= 1_048_576, s.s_n >= 0;
}

/// An in-bounds edit lands inside its own tensor's block of the layout.
proof fn lemma_pos_bounds(s: Shapes, e: Edit)
    requires
        shapes_wf(s),
        rule3(s, e),
    ensures
        e.tensor == Q_WEIGHTS ==> 0 <= pos(s, e) < q_len(s),
        e.tensor == SYNERGY ==> q_len(s) <= pos(s, e) < q_len(s) + s_len(s),
        e.tensor == TERMINATE_WEIGHTS ==> q_len(s) + s_len(s) <= pos(s, e) < total_entries(s),
        0 <= pos(s, e) < total_entries(s),
{
    lemma_total_bound(s);
    assert(0 <= e.index@[0] < dims(s, e.tensor)[0]);
    if e.tensor != TERMINATE_WEIGHTS {
        assert(0 <= e.index@[1] < dims(s, e.tensor)[1]);
    }
    if e.tensor == Q_WEIGHTS {
        lemma_row_major(e.index@[0] as int, e.index@[1] as int, s.q_rows as int, s.q_cols as int);
    } else if e.tensor == SYNERGY {
        lemma_row_major(e.index@[0] as int, e.index@[1] as int, s.s_n as int, s.s_n as int);
    }
}

proof fn lemma_row_major(i: int, j: int, rows: int, cols: int)
    requires
        0 <= i < rows,
        0 <= j < cols,
    ensures
        0 <= i * cols + j < rows * cols,
{
    assert(0 <= i * cols + j < rows * cols) by (nonlinear_arith)
        requires 0 <= i < rows, 0 <= j < cols;
}

/// Two in-bounds edits that land on the same entry have the same target.
proof fn lemma_pos_injective(s: Shapes, a: Edit, b: Edit)
    requires
        shapes_wf(s),
        rule3(s, a),
        rule3(s, b),
        pos(s, a) == pos(s, b),
    ensures
        same_target(a, b),
{
    lemma_pos_bounds(s, a);
    lemma_pos_bounds(s, b);
    assert(a.tensor == b.tensor);
    let ia = a.index@;
    let ib = b.index@;
    assert(0 <= ia[0] < dims(s, a.tensor)[0]);
    assert(0 <= ib[0] < dims(s, b.tensor)[0]);
    if a.tensor != TERMINATE_WEIGHTS {
        assert(0 <= ia[1] < dims(s, a.tensor)[1]);
        assert(0 <= ib[1] < dims(s, b.tensor)[1]);
        let cols = dims(s, a.tensor)[1];
        let x = ia[0] * cols + ia[1];
        assert(x == ib[0] * cols + ib[1]);
        lemma_fundamental_div_mod_converse(x, cols, ia[0] as int, ia[1] as int);
        lemma_fundamental_div_mod_converse(x, cols, ib[0] as int, ib[1] as int);
        assert(ia[0] == ib[0] && ia[1] == ib[1]);
    }
    assert(ia =~= ib);
}

// ─── The guard ──────────────────────────────────────────────────────────────

/// Check a weight patch against R1 to R9, rule by rule; the first rule broken
/// rejects it.
pub fn check_patch(s: &Shapes, cur: &Vec<i64>, lim: &Limits, edits: &Vec<Edit>) -> (v: Verdict)
    requires
        guard_wf(*s, cur@, *lim, edits@),
    ensures
        // W1. An accepted patch satisfies every rule, R1 to R9.
        v.rule == 0 ==> holds_through(9, *s, cur@, *lim, edits@), // W1
        // W2. A rejected patch is rejected by a rule it breaks, and by the first such rule: every earlier rule holds.
        v.rule != 0 ==> 1 <= v.rule <= 9 && !rule_holds(v.rule as int, *s, cur@, *lim, edits@) && holds_through(v.rule - 1, *s, cur@, *lim, edits@), // W2
        // W3. For a per-edit rule, the verdict names the first edit that breaks it: it breaks the rule, and every earlier edit keeps it.
        2 <= v.rule <= 7 ==> v.edit < edits@.len() && !edit_passes(v.rule as int, *s, cur@, *lim, edits@[v.edit as int]), // W3
        2 <= v.rule <= 7 ==> forall|j: int| 0 <= j < v.edit ==> edit_passes(v.rule as int, *s, cur@, *lim, #[trigger] edits@[j]), // W3
        // W3. For R8, the first edit that repeats an earlier target: no two edits before it share one.
        v.rule == 8 ==> v.edit < edits@.len() && exists|k: int| 0 <= k < v.edit && same_target(#[trigger] edits@[k], edits@[v.edit as int]), // W3
        v.rule == 8 ==> forall|a: int, b: int| 0 <= a < b < v.edit ==> !same_target(#[trigger] edits@[a], #[trigger] edits@[b]), // W3
{
    if edits.len() == 0 {
        return Verdict { rule: 1, edit: 0 };
    }
    let mut r: u8 = 2;
    while r <= 7
        invariant
            2 <= r <= 8,
            guard_wf(*s, cur@, *lim, edits@),
            holds_through(r - 1, *s, cur@, *lim, edits@),
        decreases 8 - r,
    {
        match first_failing(r, s, cur, lim, edits) {
            Some(k) => {
                return Verdict { rule: r, edit: k };
            },
            None => {},
        }
        proof {
            assert forall|q: int| 1 <= q <= r implies #[trigger] rule_holds(q, *s, cur@, *lim, edits@) by {
                if q < r {
                    assert(rule_holds(q, *s, cur@, *lim, edits@));
                }
            }
        }
        r = r + 1;
    }
    match first_duplicate(edits) {
        Some(k) => {
            return Verdict { rule: 8, edit: k };
        },
        None => {},
    }
    proof {
        assert forall|q: int| 1 <= q <= 8 implies #[trigger] rule_holds(q, *s, cur@, *lim, edits@) by {
            if q < 8 {
                assert(rule_holds(q, *s, cur@, *lim, edits@));
            }
        }
    }
    if (edits.len() as u128) > change_bound_exec(s, lim) {
        return Verdict { rule: 9, edit: 0 };
    }
    proof {
        assert forall|q: int| 1 <= q <= 9 implies #[trigger] rule_holds(q, *s, cur@, *lim, edits@) by {
            if q < 9 {
                assert(rule_holds(q, *s, cur@, *lim, edits@));
            }
        }
    }
    Verdict { rule: 0, edit: 0 }
}

/// Apply a patch that passed every rule.
pub fn apply_patch(s: &Shapes, cur: &Vec<i64>, lim: &Limits, edits: &Vec<Edit>) -> (out: Vec<i64>)
    requires
        guard_wf(*s, cur@, *lim, edits@),
        holds_through(9, *s, cur@, *lim, edits@),
    ensures
        // W4. Applying an accepted patch keeps every tensor's shape: the layout keeps its length.
        out@.len() == cur@.len(), // W4
        // W5. Only entries the patch targets change ...
        forall|p: int| 0 <= p < out@.len() && #[trigger] out@[p] != cur@[p] ==> targets(*s, edits@).contains(p), // W5
        // W5. ... so at most the R9 bound of entries change.
        changed(cur@, out@).len() <= change_bound(*s, *lim), // W5
        // W6. Every edit lands: the entry it targets holds its value.
        forall|k: int| 0 <= k < edits@.len() ==> out@[pos(*s, #[trigger] edits@[k])] == edits@[k].value, // W6
{
    assert(rule_holds(3, *s, cur@, *lim, edits@));
    assert(rule_holds(8, *s, cur@, *lim, edits@));
    assert(rule_holds(9, *s, cur@, *lim, edits@));
    let mut out: Vec<i64> = Vec::new();
    let mut p: usize = 0;
    while p < cur.len()
        invariant
            p <= cur@.len(),
            out@ =~= cur@.subrange(0, p as int),
        decreases cur@.len() - p,
    {
        out.push(cur[p]);
        p = p + 1;
    }
    assert(out@ =~= cur@);
    let mut k: usize = 0;
    while k < edits.len()
        invariant
            k <= edits@.len(),
            guard_wf(*s, cur@, *lim, edits@),
            rule_holds(3, *s, cur@, *lim, edits@),
            rule8(edits@),
            out@.len() == cur@.len(),
            forall|q: int| 0 <= q < out@.len() && out@[q] != cur@[q] ==> exists|j: int| 0 <= j < k && pos(*s, #[trigger] edits@[j]) == q,
            forall|j: int| 0 <= j < k ==> out@[pos(*s, #[trigger] edits@[j])] == edits@[j].value,
        decreases edits@.len() - k,
    {
        assert(edit_passes(3, *s, cur@, *lim, edits@[k as int]));
        let at = position(s, &edits[k]);
        let ghost before = out@;
        out.set(at, edits[k].value);
        proof {
            assert forall|j: int| 0 <= j < k + 1 implies out@[pos(*s, #[trigger] edits@[j])] == edits@[j].value by {
                if j < k {
                    assert(edit_passes(3, *s, cur@, *lim, edits@[j]));
                    if pos(*s, edits@[j]) == at as int {
                        lemma_pos_injective(*s, edits@[j], edits@[k as int]);
                        assert(!same_target(edits@[j], edits@[k as int]));
                    }
                }
            }
            assert forall|q: int| 0 <= q < out@.len() && out@[q] != cur@[q] implies exists|j: int| 0 <= j < k + 1 && pos(*s, #[trigger] edits@[j]) == q by {
                if q == at as int {
                    assert(pos(*s, edits@[k as int]) == q);
                } else {
                    assert(before[q] == out@[q]);
                }
            }
        }
        k = k + 1;
    }
    proof {
        broadcast use vstd::set::lemma_set_filter;
        vstd::set_lib::lemma_int_range(0, cur@.len() as int);
        let t = targets(*s, edits@);
        assert forall|q: int| 0 <= q < out@.len() && #[trigger] out@[q] != cur@[q] implies t.contains(q) by {
            let j = choose|j: int| 0 <= j < edits@.len() && pos(*s, #[trigger] edits@[j]) == q;
            assert(t[j] == q);
        }
        assert forall|q: int| #[trigger] changed(cur@, out@).contains(q) implies t.to_set().contains(q) by {
            assert(0 <= q < out@.len() && out@[q] != cur@[q]);
            assert(t.contains(q));
        }
        assert(changed(cur@, out@).subset_of(t.to_set()));
        t.lemma_cardinality_of_set();
        vstd::set_lib::lemma_len_subset(changed(cur@, out@), t.to_set());
    }
    out
}

} // verus!

// ─── The trusted quantizer (TB2 in the README, not verified) ────────────────

/// round(x * 1e6), half away from zero, saturating at ±MICRO_MAX; NaN and the
/// infinities are NONFINITE. wsp_guard.py's `quantize` is the same function.
pub fn quantize(x: f64) -> i64 {
    if !x.is_finite() {
        return NONFINITE;
    }
    let r = (x * 1e6).round();
    if r >= MICRO_MAX as f64 {
        MICRO_MAX
    } else if r <= -(MICRO_MAX as f64) {
        -MICRO_MAX
    } else {
        r as i64
    }
}
