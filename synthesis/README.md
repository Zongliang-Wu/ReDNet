# Data Synthesis Pipeline

Synthesizes (dirty, clean, dirt-mask) triplets following the data
construction of the paper (Sec. III): clean patches from DIV2K / Flickr2K
are composited with dirt patterns under randomly sampled attenuation,
scattering and illumination conditions, so that dirt appears dark against
bright backgrounds and bright against dark backgrounds, as observed in real
dirty-window photographs.

`dust_syn.py` is a vectorized port of the original research implementation.

## Pipeline

```
DIV2K / Flickr2K images ──prepare_clean.py──▶ 256² clean patches
                                                     │
raw dirt patterns ───────prepare_patterns.py──▶ 256² pattern masks ──┐
  (Gu et al. 2007 dust synthesis tool;                                │
   self-collected whiteboard captures)                                ▼
                                               generate_dataset.py ──▶ gt / dirty / mask
```

## Usage

```bash
# 1) clean patches (Flickr2K 0001-2000 for train, 2001+ for the 100-image test)
python synthesis/prepare_clean.py --input_dir /path/to/Flickr2K_HR --output_dir work/clean_train --size 256

# 2) pattern masks from a raw pattern library
python synthesis/prepare_patterns.py --pattern_dir datasets/dirt_patterns_self_collected \
    --output_dir work/patterns_256 --crop 512 --size 256

# 3) triplets
python synthesis/generate_dataset.py --clean_dir work/clean_train \
    --pattern_dir work/patterns_256 --output_dir work/syn_train --n 4000 --seed 100
```

The generated folder layout (`gt/ dirty/ mask/`) matches what
`options/train_rednet.yml` expects. To regenerate data closest to the
released training distribution, point `--pattern_dir` at the shipped mask
folder (e.g. `datasets/train/mask` from the Hugging Face distribution).

## Dirt pattern sources

The **paper's pattern library is included** in the Hugging Face dataset as
`datasets/dirt_patterns_256/` (256×256 grayscale masks, white = dirt):

- `train/` — 3962 patterns used for the 4000 training samples (Table I);
- `test/` — 815 patterns, source pool of the 100-image synthetic test set.

They combine patterns rendered with the *dust synthesis tool* of
[Gu et al., "Dirty Glass: Rendering Contamination on Transparent Surfaces"
(2007)](https://dl.acm.org/doi/10.1145/1276377.1276412) and **self-collected
patterns** (`datasets/dirt_patterns_self_collected`, 74 raw images:
whiteboard-capture photos processed by color conversion, vignetting
correction and inversion). The library is distributed strictly for academic
research with full attribution to the original authors; contact us if you
are a copyright holder and want the derived patterns removed. When extending
the library with new captures, use `prepare_patterns.py` (it auto-inverts
whiteboard-style photos so that dirt stays white on black). Please respect
the license of the original Gu et al. tool if you re-render its patterns.

## Notes

- The released `datasets/train` (gated HF distribution) is the exact training
  set used for the paper's models; regenerating with this pipeline yields an
  equivalent (randomly re-sampled) dataset, not a bit-identical copy.
- Real-world capture preprocessing (dirty/GT pair registration, NR image
  preparation) is camera- and scene-specific; see the paper (Sec. III-C) for
  the protocol (identical position & camera settings, 10-frame averaging,
  ROI crop, resize to 1200×900).
