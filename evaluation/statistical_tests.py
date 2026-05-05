"""
SHARED STATISTICAL SIGNIFICANCE TESTS
Wilcoxon signed-rank test for paired comparisons between model configs.

USAGE:
    from evaluation.statistical_tests import wilcoxon_test, pairwise_wilcoxon

    result = wilcoxon_test([0.91, 0.92, 0.90], [0.88, 0.87, 0.86])
    pairs, summary = pairwise_wilcoxon({"ModelA": [0.91, 0.92, 0.90],
                                         "ModelB": [0.88, 0.87, 0.86]})
"""
from __future__ import annotations

import csv
import logging
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

_MIN_SAMPLES_WARNING = 6


def wilcoxon_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> Dict[str, float | bool | str]:
    """Wilcoxon signed-rank test for paired samples.

    Args:
        scores_a: metric values for condition A (one per seed/fold).
        scores_b: metric values for condition B (same length).
        alternative: 'two-sided', 'greater', or 'less'.
        alpha: significance threshold.

    Returns dict with:
        statistic:    Wilcoxon W statistic
        p_value:      two-sided (or one-sided) p-value
        significant:  True if p_value < alpha
        effect_size:  rank-biserial correlation r = 1 - (2W / n(n+1))
        n_pairs:      number of paired observations
        note:         warning string if n is very small
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)

    if len(a) != len(b):
        raise ValueError(f"Unequal lengths: {len(a)} vs {len(b)}")

    n = len(a)
    note = ""

    if n < 2:
        logger.warning("Need at least 2 paired observations for Wilcoxon test.")
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "significant": False,
            "effect_size": float("nan"),
            "n_pairs": n,
            "note": "Insufficient samples (n < 2)",
        }

    if n < _MIN_SAMPLES_WARNING:
        note = (
            f"Small sample size (n={n}); Wilcoxon has limited power. "
            f"Minimum achievable p-value for n={n} is {_min_p(n):.4f}."
        )
        logger.warning(note)

    diffs = a - b
    if np.all(diffs == 0):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "effect_size": 0.0,
            "n_pairs": n,
            "note": "All differences are zero — identical results.",
        }

    try:
        result = stats.wilcoxon(a, b, alternative=alternative)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    except ValueError as e:
        logger.warning(f"Wilcoxon test failed: {e}")
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "significant": False,
            "effect_size": float("nan"),
            "n_pairs": n,
            "note": str(e),
        }

    n_nonzero = np.sum(diffs != 0)
    if n_nonzero > 0:
        r = 1.0 - (2.0 * statistic) / (n_nonzero * (n_nonzero + 1))
    else:
        r = 0.0

    return {
        "statistic": statistic,
        "p_value": p_value,
        "significant": p_value < alpha,
        "effect_size": round(r, 4),
        "n_pairs": n,
        "note": note,
    }


def pairwise_wilcoxon(
    results_dict: Dict[str, List[float]],
    metric: str = "auroc",
    alpha: float = 0.05,
) -> Tuple[List[Dict], str]:
    """Run Wilcoxon signed-rank between all pairs of models/configs.

    Args:
        results_dict: {model_name: [seed1_val, seed2_val, ...]}.
        metric: name of the metric (for display only).
        alpha: significance threshold.

    Returns:
        (pairs_results, summary_string)
        pairs_results: list of dicts, one per pair.
        summary_string: formatted text table.
    """
    names = sorted(results_dict.keys())
    pairs_results: List[Dict] = []

    for name_a, name_b in combinations(names, 2):
        test = wilcoxon_test(
            results_dict[name_a],
            results_dict[name_b],
            alpha=alpha,
        )
        test["model_a"] = name_a
        test["model_b"] = name_b
        test["metric"] = metric
        mean_a = float(np.mean(results_dict[name_a]))
        mean_b = float(np.mean(results_dict[name_b]))
        test["mean_a"] = round(mean_a, 4)
        test["mean_b"] = round(mean_b, 4)
        test["diff"] = round(mean_a - mean_b, 4)
        pairs_results.append(test)

    summary = significance_table(pairs_results, metric)
    return pairs_results, summary


def significance_table(
    pairs_results: List[Dict],
    metric: str = "auroc",
) -> str:
    """Format pairwise Wilcoxon results into a readable text table."""
    if not pairs_results:
        return "No pairwise comparisons to display."

    header = (
        f"{'Pair':<40s} {'Mean A':>8s} {'Mean B':>8s} "
        f"{'Diff':>8s} {'W':>8s} {'p-value':>10s} {'Sig?':>5s} {'r':>7s}"
    )
    sep = "-" * len(header)
    lines = [f"Wilcoxon Signed-Rank Tests — {metric.upper()}", sep, header, sep]

    for r in pairs_results:
        sig_flag = " *" if r.get("significant") else "  "
        p_str = f"{r['p_value']:.4f}" if not np.isnan(r["p_value"]) else "   N/A"
        w_str = f"{r['statistic']:.1f}" if not np.isnan(r["statistic"]) else "  N/A"
        r_str = f"{r['effect_size']:.4f}" if not np.isnan(r["effect_size"]) else "  N/A"
        pair_name = f"{r['model_a']} vs {r['model_b']}"
        lines.append(
            f"{pair_name:<40s} {r['mean_a']:>8.4f} {r['mean_b']:>8.4f} "
            f"{r['diff']:>+8.4f} {w_str:>8s} {p_str:>10s} {sig_flag:>5s} {r_str:>7s}"
        )

    lines.append(sep)
    lines.append(f"* significant at alpha = 0.05 | n = {pairs_results[0].get('n_pairs', '?')} paired observations")

    notes = [r["note"] for r in pairs_results if r.get("note")]
    if notes:
        lines.append(f"Note: {notes[0]}")

    return "\n".join(lines)


def save_significance_csv(
    pairs_results: List[Dict],
    save_path: str | Path,
) -> None:
    """Write pairwise test results to CSV."""
    if not pairs_results:
        return

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model_a", "model_b", "metric", "mean_a", "mean_b", "diff",
        "statistic", "p_value", "significant", "effect_size", "n_pairs", "note",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pairs_results)

    logger.info(f"Significance results saved to {path}")


def _min_p(n: int) -> float:
    """Approximate minimum achievable p-value for Wilcoxon with n pairs."""
    if n < 2:
        return 1.0
    try:
        max_stat = n * (n + 1) / 2
        result = stats.wilcoxon(
            np.arange(1, n + 1, dtype=float),
            np.zeros(n),
        )
        return float(result.pvalue)
    except Exception:
        return 2.0 ** (-(n - 1))
