"""
build_reference_model.py

Run this ONCE, locally (not on Vercel), to turn a reference panel
(1000 Genomes Phase 3, LD-pruned) into the small set of artifacts the web
app loads at request time.

WHY THIS VERSION IS DIFFERENT FROM A NAIVE PCA-ONLY BUILD
-----------------------------------------------------------
The original pipeline stored only per-sample PCA coordinates (ref_pcs) and
let the prediction endpoint do a k-nearest-neighbors vote over all 2,504
individual reference samples on every request. That has two problems:
  1. It's winner-take-all-ish: with a fixed k, a sample sitting between two
     clusters still gets a vote count dominated by whichever cluster has
     slightly more nearby points, rather than a smooth mixture.
  2. It recomputes the same "which population does this look like" logic
     from scratch on every request, using data (individual reference
     genotype coordinates) that doesn't change between requests.

Instead, this version precomputes, ONCE, per detailed population (e.g.
GBR, YRI, PUR -- 26 in the current 1000 Genomes panel):
  - a centroid (mean PCA coordinate of that population's members)
  - a spread (the population's own typical distance-to-centroid, i.e. how
    tight or diffuse that cluster is)
These become the only things the prediction endpoint compares a new
sample against -- a Gaussian/RBF-style similarity to 26 fixed points,
instead of a 2,504-point k-NN vote. This is both faster (fixed, tiny
amount of work per request) and statistically smoother: the per-population
spread means a naturally diffuse, admixed population (e.g. PEL, MXL, which
are known to be heavily admixed in 1000 Genomes) gets a wider "acceptance
radius" than a tight, historically isolated population (e.g. CHB, FIN),
so a sample doesn't get artificially over-confident just because it's
slightly closer to one cluster's edge than another's.

Expected inputs
----------------
1. A PLINK binary fileset: reference.bed / reference.bim / reference.fam
   (already LD-pruned -- see README for how this project's pruned file
   was produced).
2. A panel file mapping sample ID -> population label (1000 Genomes'
   integrated_call_samples.panel format):
       sample  pop     super_pop   gender
       HG00096 GBR     EUR         male

Outputs
-------
model/reference_model.npz containing:
    -- per-SNP arrays, needed to turn an uploaded raw file into a feature
       vector and project it into PCA space --
    snp_ids            (n_snps,)   rsIDs, in a fixed order
    ref_allele         (n_snps,)   reference/coding allele per SNP
    alt_allele         (n_snps,)   alternate allele per SNP
    allele_freq        (n_snps,)   reference-panel frequency of alt_allele
    pca_mean           (n_snps,)   mean used to center genotypes before PCA
    pca_components     (k, n_snps) PCA loading matrix (1/sqrt(2p(1-p))
                                    scaling already folded in, so
                                    projection is one matrix multiply)

    -- per-reference-sample arrays, kept only so the static scatter-plot
       asset (scripts/export_frontend_assets.py) can still draw the full
       reference cloud in the UI. NOT used by predict.py anymore. --
    ref_pcs            (n_samples, k)
    ref_super_pop      (n_samples,)
    ref_pop            (n_samples,)

    -- precomputed population-level statistics, the actual inputs to
       inference. This is the part that used to be recomputed implicitly,
       every request, via k-NN. Now it's computed once, here. --
    pop_codes          (n_pops,)      detailed population codes, e.g. "GBR"
    pop_centroids       (n_pops, k)   mean PCA coordinate per population
    pop_spread          (n_pops,)     mean distance-to-centroid per
                                       population (RBF bandwidth)
    pop_to_super        (n_pops,)     each pop_codes[i]'s superpopulation
    superpop_codes       (n_super,)   e.g. ["AFR","AMR","EAS","EUR","SAS"]

Dependencies (offline only -- NOT deployed):
    pip install numpy pandas scikit-learn
    (pandas_plink is NOT required -- see plink_reader.py, a small
    dependency-free PLINK .bed parser written for this project so the
    build doesn't require a compiled PLINK binary or extra wheels.)
"""

import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, ".")
from plink_reader import load_plink

# ---- config ----------------------------------------------------------
BED_PREFIX = "reference_pruned"
PANEL_FILE = "integrated_call_samples.panel"
N_COMPONENTS = 20
MAF_MIN = 0.05
OUT_PATH = "model/reference_model.npz"

AMBIGUOUS_PAIRS = {frozenset(("A", "T")), frozenset(("C", "G"))}


def is_strand_ambiguous(a0: str, a1: str) -> bool:
    return frozenset((a0, a1)) in AMBIGUOUS_PAIRS


def compute_population_statistics(ref_pcs: np.ndarray, pop: np.ndarray, super_pop: np.ndarray):
    """Precompute, once, everything predict.py needs to score a new sample
    against each reference population: each population's centroid (mean
    PCA position) and spread (mean within-population distance to that
    centroid, used later as a per-population RBF bandwidth).

    TODO(future upgrade): replace the scalar "spread" with a full
    per-population covariance matrix (Mahalanobis distance instead of
    isotropic Euclidean) if a future larger reference panel makes that
    estimate stable enough -- 1000 Genomes' smallest populations (~60-100
    samples) are borderline for reliably estimating a full 20x20
    covariance matrix, which is why this version uses a single scalar
    spread per population instead.
    """
    pop_codes = np.array(sorted(set(pop.tolist())))
    n_pops = len(pop_codes)
    k = ref_pcs.shape[1]

    pop_centroids = np.zeros((n_pops, k), dtype=np.float32)
    pop_spread = np.zeros(n_pops, dtype=np.float32)
    pop_to_super = np.empty(n_pops, dtype="<U8")

    for i, code in enumerate(pop_codes):
        mask = pop == code
        members = ref_pcs[mask]
        centroid = members.mean(axis=0)
        pop_centroids[i] = centroid
        dists = np.linalg.norm(members - centroid, axis=1)
        # floor the spread so a tiny/near-degenerate cluster doesn't
        # produce a near-zero bandwidth (which would make the RBF kernel
        # collapse to a near-hard-assignment for that population)
        pop_spread[i] = max(float(dists.mean()), 1e-3)
        pop_to_super[i] = super_pop[mask][0]

    superpop_codes = np.array(sorted(set(super_pop.tolist())))
    return pop_codes, pop_centroids, pop_spread, pop_to_super, superpop_codes


def main():
    print("Loading PLINK fileset...")
    geno, bim, fam = load_plink(BED_PREFIX)
    # geno: (n_samples, n_snps), float32, NaN = missing, dosage = count of a2 (bim col 6)

    snp_ids = bim["snp_id"].values.astype(str)
    a1 = bim["a1"].values.astype(str)  # allele NOT counted (dosage=0 side)
    a2 = bim["a2"].values.astype(str)  # allele counted by dosage
    sample_ids = fam["iid"].values.astype(str)

    print(f"Loaded {geno.shape[1]} SNPs x {geno.shape[0]} samples")

    # ---- QC ----
    freq = np.nanmean(geno, axis=0) / 2.0
    missing_rate = np.isnan(geno).mean(axis=0)

    keep = np.ones(len(snp_ids), dtype=bool)
    keep &= missing_rate < 0.02
    keep &= (freq > MAF_MIN) & (freq < 1 - MAF_MIN)
    keep &= np.array([not is_strand_ambiguous(x, y) for x, y in zip(a1, a2)])

    print(f"QC keeps {keep.sum()} / {len(keep)} SNPs")
    geno = geno[:, keep]
    snp_ids = snp_ids[keep]
    a1 = a1[keep]
    a2 = a2[keep]
    freq = freq[keep]

    # mean-impute any remaining missing genotypes with 2*freq
    nan_mask = np.isnan(geno)
    if nan_mask.any():
        fill = (2 * freq)[np.newaxis, :]
        geno = np.where(nan_mask, np.broadcast_to(fill, geno.shape), geno)

    # ---- standardize (Patterson et al. scaling used by EIGENSOFT/smartpca) ----
    p = freq
    denom = np.sqrt(2 * p * (1 - p))
    denom[denom == 0] = 1.0
    mean = 2 * p
    geno -= mean[np.newaxis, :]
    geno /= denom[np.newaxis, :]
    X = geno  # standardized in place

    # ---- PCA ----
    print(f"Fitting PCA with {N_COMPONENTS} components...")
    pca = PCA(n_components=N_COMPONENTS, svd_solver="randomized", random_state=0)
    ref_pcs = pca.fit_transform(X)
    del X

    # fold the (x-mean)/denom scaling into the components so that, at
    # inference time, projection is: pcs = (raw_dosage - pca_mean) @ pca_components.T
    pca_components = pca.components_ / denom[np.newaxis, :]
    pca_mean = mean

    # ---- attach population labels ----
    panel = pd.read_csv(PANEL_FILE, sep="\t")
    panel.columns = [c.strip() for c in panel.columns]
    panel = panel.set_index(panel.columns[0])
    super_pop = panel.loc[sample_ids, "super_pop"].values.astype(str)
    pop = panel.loc[sample_ids, "pop"].values.astype(str)

    print("Explained variance ratio (first 5 PCs):", pca.explained_variance_ratio_[:5])

    # ---- precompute population centroids / spread / superpop mapping ----
    print("Precomputing per-population centroids and spread...")
    pop_codes, pop_centroids, pop_spread, pop_to_super, superpop_codes = (
        compute_population_statistics(ref_pcs, pop, super_pop)
    )
    for code, spread, sup in zip(pop_codes, pop_spread, pop_to_super):
        n = int((pop == code).sum())
        print(f"  {code:>4} ({sup})  n={n:4d}  spread={spread:.2f}")

    print("Saving model artifacts...")
    import os
    os.makedirs("model", exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        snp_ids=np.array(snp_ids, dtype="<U32"),
        ref_allele=np.array(a1, dtype="<U8"),
        alt_allele=np.array(a2, dtype="<U8"),
        allele_freq=freq.astype(np.float32),
        pca_mean=pca_mean.astype(np.float32),
        pca_components=pca_components.astype(np.float32),
        # kept only for the static scatter-plot asset export, not used by predict.py
        ref_pcs=ref_pcs.astype(np.float32),
        ref_super_pop=np.array(super_pop, dtype="<U8"),
        ref_pop=np.array(pop, dtype="<U8"),
        # precomputed inference inputs -- the point of this refactor
        pop_codes=np.array(pop_codes, dtype="<U8"),
        pop_centroids=pop_centroids.astype(np.float32),
        pop_spread=pop_spread.astype(np.float32),
        pop_to_super=np.array(pop_to_super, dtype="<U8"),
        superpop_codes=np.array(superpop_codes, dtype="<U8"),
    )
    print(f"Wrote {OUT_PATH}")
    print(f"  {ref_pcs.shape[0]} samples, {len(snp_ids)} SNPs, {N_COMPONENTS} PCs, {len(pop_codes)} populations")


if __name__ == "__main__":
    main()
