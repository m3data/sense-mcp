# project-a: Technical Overview

## Background

Entrainment is one of the most pervasive phenomena in nature. When two pendulum clocks are mounted on the same wall, they will gradually synchronise their swings — not through any deliberate coordination, but through the subtle transmission of vibration through shared substrate. The same principle governs the firing of heart cells, the synchronisation of fireflies, the shared rhythm of musicians playing together, and the cadences that emerge in tightly coupled teams.

This project asks: what does entrainment look like in the interaction sequences of sociotechnical systems? And more specifically: can we detect it, measure it, and use those measurements to assess coherence?

## Theoretical Framework

### Entrainment as Coordination Mechanism

Conventional models of coordination emphasise explicit communication — protocols, agreements, handoffs. But much coordination in high-functioning systems happens through a different mechanism: rhythmic coupling. Agents develop shared tempos, shared cadences of action and response, and these shared rhythms allow them to operate together without constant explicit negotiation.

This is entrainment. It is not the same as synchrony in the sense of acting simultaneously. Two agents can be entrained even when one consistently leads the other by a fixed phase offset — as long as their rhythms remain coupled, they are in an entrainment relationship.

### Coherence as Systemic Property

Coherence is distinct from coordination. A system can be coordinated in the sense that its parts move together, but lack coherence if that movement has no internal logic or legibility. Coherence implies that the system's behaviour makes sense — that there is a pattern that participants can recognise, rely on, and build from.

Sustained entrainment is one pathway to coherence. When agents' rhythms are coupled, they develop mutual predictability. That predictability supports trust, reduces coordination overhead, and creates the conditions for higher-order collaboration.

### Resonance and Amplification

When entrainment achieves a certain depth, resonance emerges. Resonance is the amplification effect — patterns are not merely shared but reinforced. In music, resonance is what gives ensemble playing its distinctive quality, beyond what any individual musician could produce alone. In organisations, resonance is what distinguishes a high-functioning team from a merely coordinated one.

Resonance is double-edged. Entrained systems in resonance are more powerful — but they are also more fragile. Perturbations that break resonance can cause rapid decoherence, as the mutual reinforcement that stabilised the system suddenly reverses.

## Measurement Approach

The project uses time-series analysis of interaction logs to detect entrainment signatures. The core metrics are:

**Cross-correlation coefficient** — measures the degree to which rhythmic patterns in one agent's sequence are reflected in another's, at varying time lags. Peak correlation at lag=0 indicates synchrony; peak at non-zero lag indicates phase offset.

**Coherence stability score** — computed from the variance of entrainment event strengths across a sequence. Low variance (high stability) indicates durable coherence; high variance indicates fragile or intermittent coherence.

**Resonance score** — combines entrainment strength with event duration. Short, strong events contribute less than sustained, moderate entrainment, reflecting the theoretical primacy of durability over peak intensity.

## Current Limitations

The current implementation treats interaction sequences as univariate time series. Real interaction data is multivariate — turns have content, affect, timing, and modality. Extending the analysis to multivariate entrainment is the next major development target.

Additionally, the current detection pipeline uses fixed windows. Adaptive windowing — where the analysis window adjusts to the dominant oscillatory frequency of the data — would substantially improve sensitivity to entrainment events at multiple scales.

## Related Work

- Strogatz, S. (2003). *Sync: The Emerging Science of Spontaneous Order.*
- Clayton, M., Sager, R., & Will, U. (2005). In time with the music: The concept of entrainment and its significance for ethnomusicology.
- Konvalinka, I., & Roepstorff, A. (2012). The two-brain approach: How can mutually interactive brains teach us something about social interaction?

*Last updated: 2026-01-20*
