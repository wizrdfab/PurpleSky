# AGENTS.md (Codex-only)

This file is directed solely at the Codex LLM operating in this repository.
These are hard, non-negotiable prohibitions. No exceptions.

Evidence Discipline
- Make claims without pointing to specific code, logs, or tests
- Assume behavior from similar systems instead of verifying this one
- Treat memory of earlier context as fact without re-checking files
- Ignore or downplay evidence that conflicts with the claim
- Present "likely" outcomes as definite
- Edit files without asking for exact wording when user intent is not explicit
- Edit files without quoting back the requested wording and receiving confirmation
- Proceed when intent is unclear instead of stating "unknown" and stopping
- Proceed with changes when the user request may be mistaken or inconsistent without asking for clarification

Code and Configuration
- Read only part of a flow and infer the rest
- Skip configuration or feature flags that change runtime behavior
- Assume defaults instead of checking actual values
- Miss error-handling paths and failure modes
- Ignore versioned model artifacts or params
- Skip strict input validation and runtime assertions on critical paths

Runtime and Environment
- Assume dependency versions match training/runtime
- Ignore external API semantics, rate limits, or edge cases
- Assume network/storage is reliable
- Ignore clock skew or timezone effects
- Skip checking live state files or caches

Timing and Concurrency
- Assume event order without proof
- Ignore race conditions between WS and REST paths
- Assume idempotency where it is not guaranteed
- Ignore retry/backoff side effects
- Treat eventual consistency as immediate consistency

Data and Inputs
- Assume input data is complete/clean
- Ignore missing or synthetic bars
- Assume features exist without validation
- Skip checking schema or column changes
- Treat out-of-range values as impossible

Testing and Validation
- Skip reproducing with a minimal test
- Rely on "should work" instead of actual output
- Do not check edge cases and failure paths
- Skip verifying metrics or alerts in logs
- Do not compare against backtest or baseline outputs
- Skip creating unit tests for critical code paths
- Skip creating a unit test when a mistake is possible
- Treat money-loss risk paths as non-critical
- Skip formal verification or proofs for critical functions
- Skip exhaustive or property-based tests when the input space allows it
