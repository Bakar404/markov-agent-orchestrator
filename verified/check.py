#!/usr/bin/env python3
"""Check the verified kernel end to end, and write verified/target/receipt.json.

    1. prove      Verus proves src/lib.rs against its specification.
    2. mutants    Plant bugs in a copy of lib.rs, one or more per clause; Verus must reject each
                  one, and must reject it on the clause it breaks. A proof that survives a planted
                  bug is not checking what it claims to.
    3. build      cargo builds the kernel CLI (src/main.rs) that the differential test talks to.
    4. diff       backend/tests/test_verified_kernel.py: Hypothesis inputs, Python == Rust for
                  the gate, the budget and the guard.
    5. control    Plant bugs in the Python guard; each must make the differential test fail. A
                  differential test that passes a wrong Python is not checking anything.
    6. receipt    Versions, hashes, counts and timings, in verified/target/receipt.json.

Needs VERUS (the verus binary; default: verus on PATH), cargo and rustup with the toolchain in
rust-toolchain.toml, and the backend's Python environment with Hypothesis. Run it with that
Python: ``python verified/check.py``. Exits non-zero if any step fails.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BACKEND = ROOT / "backend"
TARGET = HERE / "target"
LOGS = TARGET / "logs"
LIB = HERE / "src" / "lib.rs"
DIFF_TEST = BACKEND / "tests" / "test_verified_kernel.py"

RESULTS = re.compile(r"verification results:: (\d+) verified, (\d+) errors?")
LOCATION = re.compile(r"-->\s+.*?lib\.rs:(\d+):\d+")
TAG = re.compile(r"// ([GBRW]\d+)\s*$")


@dataclass(frozen=True)
class Mutant:
    id: str
    clause: str  # the clause whose proof the bug must break
    bug: str
    original: str
    replacement: str


MUTANTS = [
    # 1. The escalation gate
    Mutant("M01", "G1", "specialists are legal before escalation",
           "let specialists = escalated;", "let specialists = true;"),
    Mutant("M02", "G2", "the generalist needs allow_escalation",
           "r.push(!escalated);  // invoke_generalist",
           "r.push(!escalated && g.allow_escalation);  // invoke_generalist"),
    Mutant("M03", "G3", "escalate ignores allow_escalation",
           "let escalate = !escalated && g.allow_escalation && g.solo_steps",
           "let escalate = !escalated && g.solo_steps"),
    Mutant("M04", "G4", "a run can escalate twice",
           "r.push(escalate);  // escalate", "r.push(escalate || escalated);  // escalate"),
    Mutant("M05", "G5", "invoke_memory is never legal",
           "r.push(specialists);  // invoke_memory", "r.push(false);  // invoke_memory"),
    Mutant("M06", "G6", "run_parallel before escalation",
           "let parallel = escalated && above_exec(", "let parallel = above_exec("),
    Mutant("M07", "G7", "terminate ignores min_steps_before_terminate",
           "let terminate = g.step >= g.min_steps_before_terminate && (", "let terminate = ("),
    # 2. Budget accounting
    Mutant("M08", "B1", "remaining is not floored at zero",
           "if d < 0 { 0 } else { d }", "d"),
    Mutant("M27", "B2", "an untouched budget reports more than its total",
           "if d < 0 { 0 } else { d }", "if d < 0 { 0 } else if spent == 0 { total + 1 } else { d }"),
    Mutant("M28", "B3", "remaining halves what's left",
           "if d < 0 { 0 } else { d }", "if d < 0 { 0 } else { d / 2 }"),
    Mutant("M09", "B4", "normalized exceeds 1 when nothing is spent",
           "    } else if rem >= total {\n        MICRO\n",
           "    } else if rem >= total {\n        MICRO + 1\n"),
    Mutant("M10", "B5", "normalized is not the exact ratio",
           "        q as i64\n", "        (q as i64) / 2\n"),
    Mutant("M11", "B6", "a negative cost refunds the spend",
           "let c = if cost < 0 { 0 } else { cost };", "let c = cost;"),
    Mutant("M12", "B7", "a spend that would saturate wraps to zero, refunding the budget",
           "let s = if spent > MICRO_MAX - c { MICRO_MAX } else { spent + c };",
           "let s = if spent > MICRO_MAX - c { 0 } else { spent + c };"),
    # 3. The guard: each rule
    Mutant("M13", "R2", "tensor code 3 counts as known",
           "e.tensor == Q_WEIGHTS || e.tensor == SYNERGY || e.tensor == TERMINATE_WEIGHTS",
           "e.tensor <= TERMINATE_WEIGHTS + 1"),
    Mutant("M14", "R3", "terminate_weights accepts any arity",
           "ix.len() == 1 && in_bounds(ix[0], s.t_len)",
           "ix.len() >= 1 && in_bounds(ix[0], s.t_len)"),
    Mutant("M15", "R4", "the synergy diagonal is accepted",
           "e.tensor != SYNERGY || e.index[0] < e.index[1]",
           "e.tensor != SYNERGY || e.index[0] <= e.index[1]"),
    Mutant("M16", "R5", "non-finite values are accepted",
           "        ok == rule5(*e), // R5\n{\n    e.value != NONFINITE\n",
           "        ok == rule5(*e), // R5\n{\n    true\n"),
    Mutant("M17", "R6", "|value| == max_abs_value is rejected",
           "a <= lim.max_abs\n", "a < lim.max_abs\n"),
    Mutant("M18", "R7", "the delta limit is off by a micro-unit",
           "a <= lim.max_delta\n", "a <= lim.max_delta + 1\n"),
    Mutant("M19", "R8", "duplicates are never reported",
           "                return Some(k2);\n", "                return None;\n"),
    Mutant("M20", "R9", "the change bound is one too high",
           "(lim.max_fraction as u128) * total / (MICRO as u128)",
           "(lim.max_fraction as u128) * total / (MICRO as u128) + 1"),
    # 3. The guard: the verdict and applying it
    Mutant("M21", "W1", "a one-edit patch is accepted unchecked",
           "        return Verdict { rule: 1, edit: 0 };\n    }\n",
           "        return Verdict { rule: 1, edit: 0 };\n    }\n"
           "    if edits.len() == 1 {\n        return Verdict { rule: 0, edit: 0 };\n    }\n"),
    Mutant("M22", "W2", "an R9 failure is reported as R8",
           "return Verdict { rule: 9, edit: 0 };", "return Verdict { rule: 8, edit: 0 };"),
    Mutant("M23", "W3", "the verdict names the wrong edit",
           "return Verdict { rule: r, edit: k };", "return Verdict { rule: r, edit: 0 };"),
    # Names an edit that does break the rule, so only the "first" half of W3 can catch it.
    Mutant("M29", "W3", "the verdict names the second offending edit, not the first",
           "        if !edit_passes_exec(r, s, cur, lim, &edits[k]) {\n"
           "            return Some(k);\n        }\n",
           "        if !edit_passes_exec(r, s, cur, lim, &edits[k]) {\n"
           "            let mut k2: usize = k + 1;\n"
           "            while k2 < edits.len()\n"
           "                invariant\n"
           "                    2 <= r <= 7,\n"
           "                    k < k2 <= edits@.len(),\n"
           "                    guard_wf(*s, cur@, *lim, edits@),\n"
           "                    holds_through(r - 1, *s, cur@, *lim, edits@),\n"
           "                    !edit_passes(r as int, *s, cur@, *lim, edits@[k as int]),\n"
           "                decreases edits@.len() - k2,\n"
           "            {\n"
           "                proof {\n"
           "                    let e2 = edits@[k2 as int];\n"
           "                    assert(e2.value == NONFINITE || in_range(e2.value as int));\n"
           "                    if r >= 4 {\n"
           "                        assert(rule_holds(3, *s, cur@, *lim, edits@));\n"
           "                        assert(edit_passes(3, *s, cur@, *lim, e2));\n"
           "                    }\n"
           "                    if r >= 6 {\n"
           "                        assert(rule_holds(5, *s, cur@, *lim, edits@));\n"
           "                        assert(edit_passes(5, *s, cur@, *lim, e2));\n"
           "                    }\n"
           "                }\n"
           "                if !edit_passes_exec(r, s, cur, lim, &edits[k2]) {\n"
           "                    return Some(k2);\n"
           "                }\n"
           "                k2 = k2 + 1;\n"
           "            }\n"
           "            return Some(k);\n        }\n"),
    Mutant("M24", "W4", "applying a patch grows the layout",
           "    }\n    out\n}\n\n} // verus!", "    }\n    out.push(0);\n    out\n}\n\n} // verus!"),
    Mutant("M25", "W5", "applying a patch also zeroes entry 0",
           "    }\n    out\n}\n\n} // verus!",
           "    }\n    if out.len() > 0 { out.set(0, 0); }\n    out\n}\n\n} // verus!"),
    Mutant("M26", "W6", "applying a patch skips its last edit",
           "    let mut k: usize = 0;\n    while k < edits.len()\n        invariant\n"
           "            k <= edits@.len(),\n            guard_wf(",
           "    let mut k: usize = 0;\n    while k + 1 < edits.len()\n        invariant\n"
           "            k <= edits@.len(),\n            guard_wf("),
]


# ─── Helpers ──────────────────────────────────────────────────────────────────────────────────


def now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None,
        log: str | None = None) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})},
                          capture_output=True, text=True, check=False)
    if log:
        (LOGS / log).write_text(f"$ {' '.join(cmd)}\n\n{proc.stdout}\n{proc.stderr}",
                                encoding="utf-8")
    return proc


def say(msg: str) -> None:
    print(msg, flush=True)


def toolchain_env() -> dict[str, str]:
    """Pin rustup to rust-toolchain.toml's channel, whatever directory a tool runs in."""
    text = (HERE / "rust-toolchain.toml").read_text(encoding="utf-8")
    channel = re.search(r'channel\s*=\s*"([^"]+)"', text)
    return {"RUSTUP_TOOLCHAIN": channel.group(1)} if channel else {}


def verus_binary() -> str:
    verus = os.environ.get("VERUS") or shutil.which("verus")
    if not verus or not Path(verus).is_file():
        sys.exit("check.py: set VERUS to the verus binary (or put verus on PATH)")
    return verus


def verify(verus: str, lib: Path, log: str) -> tuple[int | None, int | None, str]:
    proc = run([verus, str(lib), "--crate-type=lib", "--multiple-errors", "50"],
               cwd=HERE, env=toolchain_env(), log=log)
    out = proc.stdout + proc.stderr
    m = RESULTS.search(out)
    if not m:
        return None, None, out
    return int(m.group(1)), int(m.group(2)), out


def clauses_hit(source: str, output: str) -> set[str]:
    """The clause tags of the specification lines Verus points at.

    A failed postcondition is reported at the clause; a clause that spans several lines (a
    match, say) carries its tag on its last line, so read forward to the end of the clause."""
    lines = source.splitlines()
    hit: set[str] = set()
    for m in LOCATION.finditer(output):
        for line in lines[int(m.group(1)) - 1:int(m.group(1)) + 8]:
            tag = TAG.search(line)
            if tag:
                hit.add(tag.group(1))
                break
            if line.strip() in ("{", "") or line.lstrip().startswith(("fn ", "pub fn ", "//")):
                break
    return hit


# ─── Steps ────────────────────────────────────────────────────────────────────────────────────


def step_prove(verus: str) -> dict:
    verified, errors, out = verify(verus, LIB, "prove.log")
    ok = verified is not None and errors == 0
    say(f"prove     {'ok' if ok else 'FAILED'}: {verified} verified, {errors} errors")
    if not ok:
        say(out[-4000:])
    return {"ok": ok, "verified": verified, "errors": errors}


def apply_mutant(source: str, m: Mutant) -> str:
    if source.count(m.original) != 1:
        raise ValueError(f"{m.id}: the text to mutate appears {source.count(m.original)} times "
                         "in lib.rs (expected once); update the mutant")
    return source.replace(m.original, m.replacement)


def step_mutants(verus: str, jobs: int) -> dict:
    source = LIB.read_text(encoding="utf-8")
    tagged = {t for line in source.splitlines() for t in TAG.findall(line)}
    work = TARGET / "mutants"
    shutil.rmtree(work, ignore_errors=True)

    def one(m: Mutant) -> dict:
        try:
            mutated = apply_mutant(source, m)
        except ValueError as e:
            return {"id": m.id, "clause": m.clause, "bug": m.bug, "caught": False,
                    "on_clause": False, "why": str(e)}
        path = work / m.id / "lib.rs"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(mutated, encoding="utf-8")
        verified, errors, out = verify(verus, path, f"mutant-{m.id}.log")
        hit = clauses_hit(mutated, out)
        compiled = verified is not None
        caught = compiled and (errors or 0) > 0
        return {"id": m.id, "clause": m.clause, "bug": m.bug, "caught": caught,
                "on_clause": caught and m.clause in hit, "errors": errors,
                "clauses_failed": sorted(hit),
                "why": None if compiled else "did not compile, so it tested nothing"}

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(one, MUTANTS))
    for r in results:
        mark = "rejected" if r["on_clause"] else ("WRONG CLAUSE" if r["caught"] else "SURVIVED")
        say(f"  {r['id']} {r['clause']:<3} {mark:<12} {r['bug']}"
            + (f"  [{', '.join(r['clauses_failed'])}]" if r.get("clauses_failed") else "")
            + (f"  ({r['why']})" if r.get("why") else ""))
    killed = sum(r["on_clause"] for r in results)
    untested = sorted(tagged - {m.clause for m in MUTANTS})
    # A tagged clause with no mutant is a claim nothing has tried to break.
    ok = killed == len(MUTANTS) and not untested
    say(f"mutants   {'ok' if ok else 'FAILED'}: {killed}/{len(MUTANTS)} rejected on their "
        f"clause" + (f"; clauses with no mutant: {', '.join(untested)}" if untested else ""))
    return {"ok": ok, "planted": len(MUTANTS), "rejected_on_clause": killed,
            "clauses_without_mutant": untested, "results": results}


def step_build() -> dict:
    cargo = shutil.which("cargo")
    if cargo is None:
        say("build     FAILED: cargo not found")
        return {"ok": False, "binary": None}
    proc = run([cargo, "build", "--release", "--locked", "--bin", "verified-kernel"], cwd=HERE,
               env={**toolchain_env(), "CARGO_TARGET_DIR": str(TARGET)}, log="build.log")
    exe = TARGET / "release" / ("verified-kernel.exe" if os.name == "nt" else "verified-kernel")
    ok = proc.returncode == 0 and exe.is_file()
    say(f"build     {'ok' if ok else 'FAILED'}: {exe.relative_to(ROOT) if ok else proc.stderr[-2000:]}")
    return {"ok": ok, "binary": str(exe) if ok else None,
            "sha256": sha256(exe) if ok else None}


def pytest_diff(binary: str, log: str, extra_env: dict[str, str]) -> tuple[int, str]:
    proc = run([sys.executable, "-m", "pytest", str(DIFF_TEST.relative_to(BACKEND)), "-q",
                "-p", "no:cacheprovider"], cwd=BACKEND, log=log,
               env={"REQUIRE_VERIFIED": "1", "VERIFIED_KERNEL_BIN": binary, **extra_env})
    return proc.returncode, proc.stdout + proc.stderr


def summary_line(out: str) -> str:
    lines = [ln for ln in out.strip().splitlines() if re.search(r"\d+ (passed|failed)", ln)]
    return lines[-1].strip("= ") if lines else out.strip().splitlines()[-1:][0] if out else ""


def step_differential(binary: str) -> dict:
    results = TARGET / "differential.json"
    results.unlink(missing_ok=True)
    rc, out = pytest_diff(binary, "differential.log", {"VERIFIED_RESULTS": str(results)})
    counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|skipped|errors?)", out)}
    seen = json.loads(results.read_text(encoding="utf-8")) if results.is_file() else None
    ok = rc == 0 and counts.get("skipped", 0) == 0 and seen is not None
    say(f"diff      {'ok' if ok else 'FAILED'}: {summary_line(out)}")
    if not ok:
        say(out[-4000:])
    return {"ok": ok, "tests": counts, "exercised": seen}


def step_negative_control(binary: str) -> dict:
    ids = re.findall(r'^\s*"(PM\d+)": \(', DIFF_TEST.read_text(encoding="utf-8"), re.MULTILINE)
    results = []
    for pm in ids:
        rc, out = pytest_diff(binary, f"control-{pm}.log", {"VERIFIED_PY_MUTANT": pm})
        stale = "no longer applies" in out
        failed = [ln.split(" - ")[0].removeprefix("FAILED ") for ln in out.splitlines()
                  if ln.startswith("FAILED ")]
        detected = rc != 0 and not stale and bool(failed)
        results.append({"id": pm, "detected": detected, "failing_tests": failed,
                        "why": "the mutant no longer applies to wsp_guard.py" if stale else None})
        say(f"  {pm} {'detected' if detected else 'NOT DETECTED'}"
            + (f": {', '.join(t.split('::')[-1] for t in failed)}" if failed else ""))
    ok = bool(ids) and all(r["detected"] for r in results)
    say(f"control   {'ok' if ok else 'FAILED'}: {sum(r['detected'] for r in results)}/{len(ids)} "
        "wrong Python guards fail the differential test")
    return {"ok": ok, "mutants": len(ids), "detected": sum(r["detected"] for r in results),
            "results": results}


# ─── Main ─────────────────────────────────────────────────────────────────────────────────────


def tool_version(cmd: list[str], cwd: Path = HERE) -> str | None:
    try:
        proc = run(cmd, cwd=cwd, env=toolchain_env())
    except OSError:
        return None
    return (proc.stdout or proc.stderr).strip() or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)),
                        help="Verus runs in parallel for the mutation check")
    parser.add_argument("--skip-mutants", action="store_true",
                        help="skip the mutation check (the receipt records that it was skipped)")
    args = parser.parse_args()

    TARGET.mkdir(exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    verus = verus_binary()
    started, t0 = now(), time.monotonic()
    steps: dict[str, dict] = {}
    timings: dict[str, float] = {}

    def timed(name: str, fn, *a):
        t = time.monotonic()
        steps[name] = fn(*a)
        timings[name] = round(time.monotonic() - t, 2)
        return steps[name]

    timed("prove", step_prove, verus)
    if args.skip_mutants:
        steps["mutants"] = {"ok": False, "skipped": True}
        say("mutants   SKIPPED")
    else:
        timed("mutants", step_mutants, verus, args.jobs)
    build = timed("build", step_build)
    if build["ok"]:
        timed("differential", step_differential, build["binary"])
        timed("negative_control", step_negative_control, build["binary"])
    ok = all(s.get("ok") for s in steps.values()) and len(steps) == 5

    try:
        import hypothesis
        hypothesis_version = hypothesis.__version__
    except ImportError:
        hypothesis_version = None
    verus_version = tool_version([verus, "--version"]) or ""
    m = re.search(r"Version:\s*(\S+)", verus_version)
    receipt = {
        "ok": ok,
        "started": started,
        "finished": now(),
        "seconds": round(time.monotonic() - t0, 2),
        "step_seconds": timings,
        "verus": m.group(1) if m else verus_version,
        "rustc": tool_version(["rustc", "--version"]),
        "python": platform.python_version(),
        "hypothesis": hypothesis_version,
        "platform": f"{platform.system()}-{platform.machine()}",
        "git_commit": tool_version(["git", "rev-parse", "HEAD"], ROOT),
        "sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (
            LIB, HERE / "src" / "main.rs", HERE / "Cargo.lock",
            BACKEND / "app" / "orchestration" / "wsp_guard.py", DIFF_TEST)},
        "steps": steps,
    }
    (TARGET / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    say(f"receipt   {(TARGET / 'receipt.json').relative_to(ROOT)}: {'PASS' if ok else 'FAIL'}")

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        p, mu = steps["prove"], steps.get("mutants", {})
        d, c = steps.get("differential", {}), steps.get("negative_control", {})
        with open(summary, "a", encoding="utf-8") as f:
            f.write(
                f"## Verified kernel: {'PASS' if ok else 'FAIL'}\n\n"
                f"| Step | Result |\n|---|---|\n"
                f"| Verus {receipt['verus']} | {p.get('verified')} verified, "
                f"{p.get('errors')} errors |\n"
                f"| Mutation check | {mu.get('rejected_on_clause')}/{mu.get('planted')} "
                f"planted bugs rejected on their clause |\n"
                f"| Differential (Python == Rust) | {d.get('tests')} |\n"
                f"| Negative control | {c.get('detected')}/{c.get('mutants')} wrong Python "
                f"guards detected |\n"
            )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
