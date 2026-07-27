# Local canonical masking code

Date: 2026-07-27

These are the canonical originals from this local machine, preserved verbatim for reconciliation with automatically rebuilt SageMaker code.

| Versioned file | SHA-256 | SageMaker counterpart | Notes |
| --- | --- | --- | --- |
| `sync/masking/lexical_controls_original.py` | `4c47c1d509e7c189a2a2529b4f8a670d0e4d3183ba7cc96dc50765a98b2ffac1` | `lexical_controls_rebuilt.py` | Original local driver for echo and masking controls. |
| `sync/masking/masking_lexicon_original.py` | `89eb8f33b21770025927fe78221a7d005d6d4e158c3029a7b6881bc932a00d85` | `masking_lexicon_rebuilt.py` | Historical original local lexicon. |
| `scripts/analysis/check_null_floors.py` | `b79d78f434e7fe9e6696ce25aaef58c22103e8a2303bb7af146e2daa514e1a44` | No counterpart identified | Compares analytic centroid-null floors with empirical permutation means. |

ATENÇÃO: masking_lexicon_original.py é o léxico SEM as guardas de janela-de-pessoa para preto/branco e SEM o bloqueio de triagem para amarelo. Essas guardas foram desenvolvidas e validadas no SageMaker (masking_lexicon_rebuilt.py) e NÃO devem ser perdidas. Este arquivo é referência histórica, não substituto.

`src/rbias/metrics.py` (GitHub) and `compute_scores.py` (local) are equivalent, as verified by SageMaker on 2026-07-27 with divergence `3.6e-09`.

## Not found locally

- Item D: pairwise-margin analysis script.
- Item E: dose-response analysis script.
- Item F1: amarelo sensitivity analysis script.

