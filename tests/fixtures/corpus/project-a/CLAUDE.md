# CLAUDE.md — project-a

## Project Identity

project-a is a research and implementation project focused on **coherence and entrainment** in distributed sociotechnical systems. The core hypothesis is that entrainment — the synchronisation of rhythms across loosely coupled agents — is a primary mechanism by which collective coherence emerges and is sustained.

This project builds computational tools for detecting entrainment signatures in interaction data, and theoretical frameworks for interpreting what those signatures mean.

## Key Concepts

**Entrainment** is the process by which one oscillating system influences another until they lock into a shared rhythm. In biology, this is how pacemaker cells coordinate. In music, it is how performers synchronise without explicit negotiation. In organisations, it is how teams develop shared cadences of work, communication, and decision-making.

**Coherence** is the downstream effect of sustained entrainment. A coherent system is not merely coordinated — it has an internal logic that makes its behaviour legible to participants and observers. Coherence is not rigidity; highly coherent systems can respond fluidly to perturbation precisely because their internal synchrony gives them a stable reference point.

**Resonance** refers to the amplification that occurs when entrainment is achieved. Systems in resonance reinforce each other's patterns, which can be generative or destabilising depending on the nature of the patterns involved.

## Architecture Notes

The computational core is built around time-series analysis of interaction logs. The primary pipeline:

1. Ingest raw interaction sequences (conversation turns, commit cadences, meeting rhythms)
2. Extract oscillatory features using windowed FFT and recurrence quantification
3. Compute cross-correlation between agent sequences to detect synchrony
4. Apply threshold detection to identify entrainment events
5. Annotate events with contextual metadata

The analysis pipeline is in `server.py`. Configuration is in `sense.toml`. Documentation is in `docs/`.

## Conventions

- All analysis functions return structured result objects, never raw primitives
- Entrainment detection thresholds are configurable — do not hardcode
- When adding new oscillatory features, document the theoretical basis in `docs/overview.md`
- Test data lives in `tests/fixtures/` — use realistic synthetic sequences, not random noise

## Research Questions This Project Addresses

- How does entrainment manifest in human-AI interaction sequences?
- What is the relationship between entrainment strength and coherence durability?
- Under what conditions does entrainment break down, and what are the early warning signals?
- Can entrainment detection be used to support real-time coherence monitoring?

*Session mode: Research. All significant findings should be traced.*
