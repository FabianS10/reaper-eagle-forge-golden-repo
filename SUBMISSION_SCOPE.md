# Submission Scope

This repository is a synthetic Forge ML target repo. It is not intended to compete as a standalone model benchmark.

## In scope

- Static detection of CUDA/NVIDIA-shaped assumptions.
- Static detection of weak benchmark discipline.
- Demonstration of a safer portable PyTorch benchmark path.
- Replay-style evidence manifest examples.
- Clear distinction between live benchmark claims and replay examples.

## Out of scope

- Proving AMD MI300X is faster than another GPU.
- Executing arbitrary repository code during Forge Repo Scan mode.
- Full CUDA-to-ROCm automatic migration.
- Security vulnerability remediation.
- Production inference serving.

## Why this matters for Forge

A migration tool can generate suggestions. Forge should do something stricter: identify what the team is allowed to claim from the available evidence.

The default claim for this synthetic repository is:

> The repository contains both NVIDIA-shaped benchmark-theater risk signals and a cleaner portable benchmark path, allowing Forge to demonstrate readiness scoring, evidence classification, and claim discipline.

A stronger performance claim requires live controlled runs, repeated trials, environment provenance, SHA-256 evidence manifests, and explicit replay-versus-live labels.
