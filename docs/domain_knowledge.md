# Domain knowledge: provenance of the unit and range checks

The three domain checks (`unit_valid`, `range_plausible`, `operator_consistent`)
are what distinguish Permit-to-Proof's verification layer from general purpose
source grounding. This document records where the encoded knowledge comes from
and, just as importantly, what it does **not** claim. The data lives in
`app/core/verify.py` (`_PARAMETERS`, `_UNIT_ALIASES`, the operator cue lists).

## What the checks assert

* `unit_valid` confirms the reported unit is dimensionally appropriate for the
  parameter and its medium (air vs water). It catches a water unit on an air
  pollutant (for example `mg/L` reported for NOx) and other category errors.
* `range_plausible` confirms the numeric value sits inside a generous
  plausibility envelope for that parameter and unit. It is a sanity bound, not a
  regulatory limit: it exists to catch obvious nonsense (negatives, order of
  magnitude errors, a value far outside anything a real permit would set),
  while never flagging a legitimate limit.
* `operator_consistent` confirms the comparison operator agrees with the clause
  wording ("shall not exceed" implies a maximum, "shall not fall below" implies
  a minimum).

## Provenance of the parameter set

The parameters are the pollutants that appear routinely in two U.S. permit
programs:

* **Title V / New Source Performance Standards (air)** — NOx, SO2, CO,
  particulate matter (PM/PM10/PM2.5), VOC, and opacity. Emission standards and
  the customary units (ppm corrected to a reference O2, lb/MMBtu, gr/dscf,
  lb/hr, tons/yr, percent opacity) follow the structure of 40 CFR Part 60 (NSPS)
  and the Title V operating permit program (40 CFR Parts 70 and 71).
* **NPDES (water)** — pH, BOD5/CBOD5, TSS, dissolved oxygen, temperature, flow,
  fecal coliform, ammonia/nitrogen, total phosphorus, and oil and grease. The
  customary units (mg/L, s.u. for pH, deg C, MGD, CFU/100 mL) and the existence
  of monthly average and weekly limits follow the NPDES program (40 CFR Part
  122) and the secondary treatment regulation (40 CFR Part 133).

## How the plausibility ranges were set

The ranges in `_PARAMETERS` are **plausibility envelopes**, deliberately
generous. They were chosen so that:

1. every legitimate permit limit for that parameter and unit falls comfortably
   inside the envelope (no false flags on real limits), and
2. values that are physically impossible (negatives), or off by orders of
   magnitude, or in the wrong medium fall outside it.

For example, NOx in ppm is bounded at `[0, 2000]`: real Title V NOx limits are
tens to low hundreds of ppm, so 2000 is a safe ceiling that still flags a
fabricated 99999 ppm. pH is bounded at `[0, 14]` (the definitional range);
opacity at `[0, 100]` percent.

## What these checks are NOT

* They are **not** the enforceable regulatory limits. A value passing
  `range_plausible` is plausible, not compliant; checking compliance against the
  specific limit in the permit is a separate, downstream task.
* The envelopes are engineering judgment calibrated to catch extraction errors,
  not a regulatory citation. A reviewer or operator can tighten them per program
  or per state without changing the verification logic.
* When a parameter is not in the table, `unit_valid` and `range_plausible` pass
  with an explicit "not checked" message rather than guessing — the layer never
  fabricates a judgment it cannot ground.

## Extending the table

Add a `ParameterSpec` to `_PARAMETERS` with the parameter's medium, its allowed
normalized units, and per-unit ranges, plus any surface-form aliases. Add unit
aliases to `_UNIT_ALIASES`. The verifier and the evaluation matcher both read
this one table, so they stay consistent automatically.
