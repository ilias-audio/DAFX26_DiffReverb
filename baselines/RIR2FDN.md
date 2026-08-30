# RIR2FDN baseline

RIR2FDN (Dal Santo, De Bortoli, Schlecht, "RIR2FDN: Modelling Room Impulse
Response Reverberation with a Feedback Delay Network", DAFx 2024) is one of the
three baselines in the paper. Unlike NoiseShaper and DFDN, which are pure Python
and run from this repo, its scattering-FDN configuration cannot be rendered in
Python: the upstream code refuses, and the second stage renders the impulse
response in MATLAB. That pipeline is not vendored here.

The rendered impulse responses are included, one per room:

```
docs/ir_wav/<room>/RIR2FDN.wav
```

so you can compare them against the reference and the fitted FDN directly. They
are also on the listening page and as MP3 under `docs/audio/<room>/RIR2FDN/`.

To reproduce them, use the upstream repositories:

- https://github.com/gdalsanto/rir2fdn
- https://github.com/georg-goetz/DecayFitNet (decay estimation)

Phase A (Python) does EDC estimation with DecayFitNet plus the colorless
scattering-FDN parameter solve; phase B (MATLAB R2024b) renders the impulse
response. The configuration used in the paper is a 6-line scattering FDN with
delays `[593, 743, 929, 1153, 1399, 1699]`, a two-stage attenuation filter and
GEQ tone correction, with EDC parameters from DecayFitNet. Peak-normalise the
rendered IR at 48 kHz and score it against `docs/ir_wav/<room>/reference.wav`
with the same `reverb.metrics.compare_rir_metrics` that `fit.py` uses.
