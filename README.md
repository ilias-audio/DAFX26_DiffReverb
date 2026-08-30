# DAFX26_DiffReverb

Code and audio examples for:

Ilias Ibnyahya and Joshua D. Reiss, "Gradient Descent Optimization of Room
Impulse Responses with Parameter-Efficient Differentiable Feedback Delay
Networks", DAFx 2026.

A 16-line FDN at 48 kHz is fitted to a measured room impulse response by
gradient descent. Delay lengths, feedback matrix, early-reflection taps and PEQ attenuation filter are trained together.

Listening page: https://ilias-audio.github.io/DAFX26_DiffReverb/

## Install

Python 3.11. GPU recommended, CPU works.

```bash
git clone https://github.com/ilias-audio/DAFX26_DiffReverb
cd DAFX26_DiffReverb
git submodule update --init
pip install -r requirements.txt
python smoke_test.py          # ~30 s CPU check that the install works
```

The `flamo/` submodule is a patched fork and the code will not run against a
pip-installed `flamo` (see "flamo fork" below).

Make sure torch actually sees your GPU. `requirements.txt` does not pin a torch
build, so pip takes the newest one, which may want a newer NVIDIA driver than you
have. If `torch.cuda.is_available()` is False, install the build for your CUDA
version:

```bash
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

Checked on a clean Python 3.11 install with torch 2.8.0+cu128.

## Fit a RIR

```bash
python fit.py --target your_room.wav                 # trainable delays
python fit.py --target your_room.wav --fixed-delays  # fixed delays, train the rest
```

Takes about two minutes per RIR on one GPU. Results go to `--out` (default
`output/<timestamp>_fit/`): `ir_target.wav`, `ir_init.wav`, `ir_optim.wav`,
`loss_history.csv`, `comparison.png` and `metrics/metrics_comparison.csv`.
Check that CSV against [EXPECTED.md](EXPECTED.md).

The nine targets used in the paper are in `docs/ir_wav/<room>/reference.wav`:

```bash
for r in docs/ir_wav/*/reference.wav; do
  python fit.py --target "$r" --out "output/$(basename $(dirname $r))"
done
```

## Code

- `model.py`: the FDN. 16 delay lines, orthogonal feedback matrix (Hadamard
  warm start), shared PEQ attenuation, K trainable early-reflection taps,
  post-sum tone PEQ and bandpass. Rendered in closed form (impulse, FFT, FDN,
  iFFT). Architecture figure: `docs/assets/architecture.png`.
- `fit.py`: training loop, metrics and plots.
- `reverb/`: losses, metrics and the Lightning wrapper.
- `baselines/`: `noiseshaper.py` and `dfdn_mezza.py` (Mezza et al., DAFx 2024)
  are runnable: `pip install -r baselines/requirements.txt`, then
  `python baselines/noiseshaper.py --target_rir <wav> --train_dir <dir>`.
  RIR2FDN needs MATLAB; its fitted IRs ship here and reproduction is described
  in [baselines/RIR2FDN.md](baselines/RIR2FDN.md).

## Audio examples

The listening page in `docs/` is served by GitHub Pages at the URL above (repo
Settings, Pages, deploy from branch `main`, folder `/docs`). To view it locally:

```bash
cd docs && python3 -m http.server 8000   # then open http://localhost:8000
```

Opening `index.html` as a `file://` URL will not work; the page fetches
`assets/manifest.json`.

Pick a room, then compare methods column by column. Rows are the impulse
response and six dry sources convolved with it. Six methods are shown by
default (reference, the two FDN configs, and the three baselines); a toggle adds
eight ablation variants. Switching method within a row keeps the playback
position.

Rooms are nine OpenAIR spaces, from a studio (T30 about 0.25 s) to a sports hall
(about 6 s).

```
docs/
  index.html          listening page
  assets/             UI, architecture.png, manifest.json
  audio/              320 kbps MP3: dry sources and per-room renders
  ir_wav/<room>/      48 kHz WAV: reference.wav plus the headline methods
  spectra/<room>/     per-room spectrum, spectrogram and early-time figures
```

Playback audio is 320 kbps MP3 transcoded from 48 kHz renders, with relative
levels between methods preserved so A/B comparisons are fair. Lossless WAVs of
the targets and the headline methods are in `docs/ir_wav/`; the ablation
variants are MP3 only.

## flamo fork

The `flamo/` submodule is a fork of [flamo](https://github.com/gdalsanto/flamo)
(Dal Santo, De Bortoli, Schlecht, MIT) with three numerical-stability patches
that this code depends on:

1. Biquad cascades are evaluated as `exp(sum(log(H_i)))` instead of
   `prod(B)/prod(A)`, which avoids overflow and bad gradients when many sections
   are multiplied.
2. `Recursion` takes a `solve_epsilon` and adds diagonal jitter before
   `torch.linalg.solve`. Upstream has no such argument, so it would reject this
   code outright.
3. PEQ mapping constants follow the parameter's device and dtype.

`model.py`, `fit.py` and the baselines put `flamo/` at the front of `sys.path`
and print the resolved path on startup, so the vendored copy wins over any
installed one. To see the patches:

```bash
git -C flamo remote add upstream https://github.com/gdalsanto/flamo
git -C flamo fetch upstream
git -C flamo diff upstream/main..HEAD
```

## Citation

Ilias Ibnyahya and Joshua D. Reiss. "Gradient Descent Optimization of Room
Impulse Responses with Parameter-Efficient Differentiable Feedback Delay
Networks". Proceedings of the 29th International Conference on Digital Audio Effects
(DAFx26), Cambridge, MA, USA, 1–4 September 2026.

## License

MIT, see [LICENSE](LICENSE). The vendored `flamo/` is separately MIT.
RIRs are from the OpenAIR dataset.
