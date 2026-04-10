# P3 Prototype Plan

Last updated: 2026-04-11

## Purpose

This document defines a simulator-backed `P3` prototype track.

The purpose of this track is not to declare that the branch has fully entered `P3`. The purpose is to validate whether a suffix-only acceleration design is structurally sound before real-hardware evidence exists.

## Positioning

This work should be treated as:

- a `P3-prototype`
- simulator-backed
- architecture- and interface-focused

This work should **not** be treated as:

- final `P3`
- deployment-level validation
- proof of real-hardware acceleration benefit

## Prototype Goal

Build one suffix-only prototype path that can run against the existing staged runtime and be exercised from the simulator-backed runtime loop.

By the end of the prototype, we should know:

- whether the current prefix/suffix split is usable as a backend contract
- whether suffix-only execution can be wrapped in a separate runtime interface
- whether fallback and correctness checks are practical
- whether the split is clean enough to justify deeper backend work later

## Scope

The prototype should stay narrow.

In scope:

- define a suffix-only runtime contract
- package suffix rollout behind a separate interface
- preserve a local fallback path to the current no-NPU runtime
- add minimal correctness checks between prototype and baseline outputs
- exercise the prototype from simulator-backed `vlash run`

Out of scope:

- full NPU integration
- automatic graph partitioning
- hardware-specific compiler work
- aggressive optimization for peak speed
- deployment claims beyond simulator-backed validation

## Required Inputs

The prototype assumes the following are already available:

- staged runtime interfaces
- simulator-backed baseline recommendation
- runtime stats and stage timings
- documented `P3` entry criteria

These inputs already exist in the branch.

## Deliverables

The prototype track should produce:

1. one explicit suffix runtime abstraction
2. one simulator-backed prototype execution path
3. one correctness/fallback comparison path
4. one short report describing what the prototype proved and what it did not prove

## Suggested Implementation Order

### 1. Freeze The Contract

Define the exact data boundary between prefix and suffix:

- prefix context object
- suffix inputs
- suffix outputs
- required metadata

### 2. Add A Prototype Backend Interface

Introduce a narrow suffix backend interface, for example:

- `run_suffix(prefix_ctx, rollout_state, noise, ...) -> action_chunk`

The default implementation can still call the existing local path.

### 3. Add Fallback And Comparison

For the prototype to be trustworthy, it must preserve:

- a fallback to the current staged local implementation
- a small correctness comparison mode against the baseline path

### 4. Exercise It In Simulator Runtime

Run the prototype from the existing `LIBERO` simulator path:

- first sync
- then async

The goal is only to verify that the interface holds and the runtime remains usable.

## Exit Criteria

The prototype is complete when:

- suffix-only interface exists in code
- simulator-backed runtime can exercise it
- baseline fallback remains available
- correctness comparison is possible
- the prototype report clearly states whether the contract is good enough for real `P3`

## Decision Rule After Prototype

After the prototype, choose one of three outcomes:

1. stay in `P2.5` and stop here
2. continue to formal `P3`
3. revise the split before any further backend work

The prototype should reduce ambiguity, not force a premature phase change.
