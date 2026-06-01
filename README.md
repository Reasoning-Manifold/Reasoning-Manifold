# Reasoning Manifolds

Code for the paper **"Reasoning emerges from constrained inference manifolds in large language models"** ([arXiv:2605.08142](https://arxiv.org/abs/2605.08142)).

We study reasoning in large language models as an intrinsic dynamical process by examining the evolution of internal representations during inference. We find that effective reasoning dynamics emerge within a constrained structural regime characterized by three conditions:

1. **Adequate representational expressivity** — quantified by the intrinsic dimension `D_world` of static vocabulary embeddings.
2. **Spontaneous manifold compression** — quantified by the stimulus-induced intrinsic dimension `D_stim` of inference-time trajectories.
3. **Preservation of non-degenerate information volume** — quantified by `V`, the volume of the centred trajectory matrix.

We summarise reasoning health with a single label-free diagnostic (paper Eq. 15):

```
H = log(D_world) · V / exp(ε · D_stim),   ε = 0.1
```

## Installation

```bash
git clone https://github.com/<org>/reasoning-manifolds.git
cd reasoning-manifolds
pip install -e .
```

See [Dependencies](#dependencies) below for the third-party packages this code relies on.

## Usage

### Multi-repeat pipeline (D_world / D_stim / V / H at the final layer)

```bash
reasoning-manifolds run \
    --model Qwen/Qwen3-8B \
    --dataset path/to/stimuli.jsonl \
    --config qwen3 \
    --tp 1 --dp 1 --repeats 1 \
    --output-dir results/qwen3-8b/
```

The runner spawns `tp × dp` worker processes, extracts last-token hidden states from the final transformer layer for every generation step, and emits `metrics.json` plus a per-run report containing `D_world`, `D_stim`, `V`, and `H`.

### Per-layer extraction (for layer-wise analyses)

```bash
python -m reasoning_manifolds.pipeline.layerwise \
    --model Qwen/Qwen3-8B \
    --dataset path/to/stimuli.jsonl \
    --config qwen3 \
    --output-dir results/layerwise/

python scripts/compute_layerwise_metrics.py \
    --states-dir results/layerwise/Qwen3-8B/stimuli/states \
    --output-dir results/layerwise_metrics/

python scripts/merge_layer_metrics.py \
    --root  results/layerwise_metrics/ \
    --glob  '*_metrics.csv' \
    --output results/qwen3_all_models.csv
```

## Repository layout

```
src/reasoning_manifolds/    core package
  metrics.py                  TLE-ID, information volume, H (Eq. 15)
  extract.py                  forward-hook hidden-state collector
  models.py                   HF model loader (Qwen / DeepSeek / Gemma3)
  prompts.py                  decoding configs and chat templates
  data.py                     JSONL stimulus loader
  pipeline/
    launcher.py                 multi-GPU runner
    worker.py                   per-process extractor
    aggregator.py               D_stim / V / H computation
    layerwise.py                per-sample per-layer state dump
scripts/
  compute_layerwise_metrics.py  per-layer ID/V CSV from .pt dumps
  merge_layer_metrics.py        merge per-model CSVs into one table
```

## Dependencies

Our experiments are built on the [`perceptual-manifold-geometry`](https://pypi.org/project/perceptual-manifold-geometry/) Python package, which provides geometric analysis tools for high-dimensional data manifolds including intrinsic dimension, curvature, density, and topological structure. It is installed automatically by `pip install -e .`, or directly via `pip install perceptual-manifold-geometry`.

## Citation

```bibtex
@article{ma2026reasoning,
  title   = {Reasoning emerges from constrained inference manifolds in large language models},
  author  = {Ma, Yanbiao and Luo, Fei and Zhang, Linfeng and Zhao, Chuangxin and Wang, Mingxuan
             and Wu, Yinan and Qian, Zhe and Lu, Yang and Chen, Long and Cao, Zhao
             and Hao, Xiaoshuai and Wen, Ji-Rong and Han, Jungong},
  journal = {arXiv preprint arXiv:2605.08142},
  year    = {2026}
}
```

## License

MIT — see `LICENSE`.
