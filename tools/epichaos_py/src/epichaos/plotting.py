"""Plotting functions for epiCHAOS results.

All functions accept an AnnData with scores already stored in adata.uns,
or a plain DataFrame. They return a matplotlib Figure.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from anndata import AnnData


def plot_scores(
    adata_or_df: AnnData | pd.DataFrame,
    key_added: str = "epiCHAOS",
    score: str = "het_adj",
    title: str | None = None,
    figsize: tuple[float, float] = (5, 4),
    save: str | None = None,
) -> plt.Figure:
    """Bar chart of per-group epiCHAOS scores, sorted by score (replicates R vignette).

    Parameters
    ----------
    adata_or_df:
        AnnData (uses adata.uns[key_added]) or a scores DataFrame directly.
    key_added:
        Key in adata.uns where scores are stored.
    score:
        Column to plot: "het_adj" (default) or "het_raw".
    title:
        Axes title; defaults to key_added.
    figsize:
        Figure size in inches.
    save:
        If provided, save figure to this path.

    Returns
    -------
    matplotlib Figure.
    """
    if isinstance(adata_or_df, AnnData):
        df = adata_or_df.uns[key_added].copy()
    else:
        df = adata_or_df.copy()

    # Strip "group-" prefix from state labels for readability
    df["state"] = df["state"].str.removeprefix("group-")
    df = df.sort_values(score)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(df["state"], df[score] + 0.01, alpha=0.8, width=0.6)
    ax.set_xlabel("")
    ax.set_ylabel(key_added)
    ax.set_title(title or key_added)
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")

    return fig


def plot_region_scores(
    df: pd.DataFrame,
    score: str = "het_adj",
    title: str = "epiCHAOS by region",
    figsize: tuple[float, float] = (5, 6),
    save: str | None = None,
) -> plt.Figure:
    """Horizontal scatter plot of per-region epiCHAOS scores (replicates R vignette).

    Parameters
    ----------
    df:
        Scores DataFrame with columns: state, het_adj (or het_raw).
    score:
        Column to plot.
    title, figsize, save:
        Styling options.

    Returns
    -------
    matplotlib Figure.
    """
    df = df.copy()
    df["state"] = df["state"].str.removeprefix("group-")
    df = df.sort_values(score)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(df[score] + 0.01, df["state"])
    ax.set_xlabel(score)
    ax.set_ylabel("")
    ax.set_title(title)
    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")

    return fig


def plot_diff(
    result: dict,
    title: str | None = None,
    figsize: tuple[float, float] = (5, 4),
    save: str | None = None,
) -> plt.Figure:
    """Density plot of permuted vs observed score differences from compare_groups().

    Parameters
    ----------
    result:
        Dict returned by :func:`epichaos.stats.compare_groups` or
        :func:`epichaos.stats.compute_diff_eICH`.
    title, figsize, save:
        Styling options.

    Returns
    -------
    matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(result["permuted_diffs"], bins=30, density=True, alpha=0.7, label="permuted")
    ax.axvline(result["observed_diff"], color="red", linestyle="--", label="observed")
    ax.set_xlabel("Absolute score difference")
    ax.set_ylabel("Density")
    ax.set_title(title or f"p = {result['pval']:.4f}")
    ax.legend()
    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")

    return fig
