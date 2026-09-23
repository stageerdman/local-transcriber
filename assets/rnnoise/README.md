# RNNoise model

The per-track noise filter uses ffmpeg's built-in `arnndn` filter with an
RNNoise model. The model weights are **not committed** here (third-party
weights, unverified license) — fetch them with:

```
scripts/fetch_rnnoise_model.sh
```

This downloads `bd.rnnn` (the "beguiling-drafter" model from
https://github.com/GregorR/rnnoise-models) into this directory.

If the model is missing, `src.track_separation.denoise_pcm` returns its input
unchanged, so the app still works — the noise filter is simply inactive.
