"""Weight-Space Programming (WSP) for the cooperative Markov game policy.

After "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab, 2026): a general
LLM reads a network's raw weights and edits them directly from a natural-language goal, with
no gradients. Here the network is :class:`CooperativeMarkovGamePolicy`, whose three tensors
(``q_weights``, ``synergy``, ``terminate_weights``) are small enough to print in full.

The pipeline, in order (:func:`app.orchestration.wsp.program.program` runs all of it):

1. ``calibrate`` — run the base policy on *calibration* seeds only and summarise behaviour.
   Calibration seeds come from a reserved band that evaluation runs may never use.
2. ``render`` — print the weights as labelled tables, plus those diagnostics.
3. ``editor`` — a :class:`~app.orchestration.wsp.editor.WeightEditor` turns (rendered weights,
   goal) into a :class:`Patch`.
4. ``apply`` — the guard in :mod:`app.orchestration.wsp_guard` vets the patch first; only an
   accepted patch touches a *copy* of the policy.

Importing this package pulls in only the patch contract (stdlib), so the bridge can use
``patch`` and ``editor`` without numpy or the engine.
"""

from __future__ import annotations

from .patch import TENSORS, Edit, Patch, PatchFormatError

__all__ = ["TENSORS", "Edit", "Patch", "PatchFormatError"]
