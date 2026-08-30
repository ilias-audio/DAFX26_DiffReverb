# Expected results

Reference numbers so you can tell whether a run reproduced the paper. These are
the bundled studio target with the default (trainable-delay) config:

```bash
python fit.py --target docs/ir_wav/01_studio_genesis6/reference.wav
```

About 50 s on an H100/A100, a few minutes on a mid-range GPU. Look at
`<out>/metrics/metrics_comparison.csv` and `<out>/comparison.png`.

| Metric | Band | Target | Fitted | abs err |
|--------|------|-------:|-------:|--------:|
| T30 (s)  | 500 Hz  |  0.220 |  0.253 | 0.033 |
| T30 (s)  | 1 kHz   |  0.166 |  0.230 | 0.064 |
| T30 (s)  | 2 kHz   |  0.206 |  0.219 | 0.013 |
| EDT (s)  | 1 kHz   |  0.186 |  0.150 | 0.036 |
| C80 (dB) | 1 kHz   |  25.65 |  24.50 | 1.15  |
| D50 (%)  | 1 kHz   |  96.75 |  96.95 | 0.20  |
| DRR (dB) | broadband | -5.78 | -5.75 | 0.03 |

Over the mid bands (500 Hz to 8 kHz) T30 error is a few tens of ms, as reported
in the paper. `comparison.png` should show the fitted magnitude spectrum and the
Schroeder EDC tracking the target closely.

Your numbers will not match to the last decimal. A colored noise floor is
injected during training and GPUs are not bit-deterministic, so DRR, Tmix and
single-band D50/C80 can move a few points between runs while T30, EDT and DRR
stay stable. That is expected.
