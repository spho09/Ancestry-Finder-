"""
build_reference_model.py

Run this ONCE, locally (not on Vercel), to turn a reference panel
(e.g. 1000 Genomes Phase 3, or the commonly-used HapMap3-SNP subset of it)
into a small set of artifacts that the web app can load at request time.

Expected inputs
----------------
1. A PLINK binary fileset: reference.bed / reference.bim / reference.fam
   Get one from e.g.:
     - https://www.internationalgenome.org/data  (1000 Genomes Phase 3 VCFs,
       convert to PLINK with `plink --vcf ... --make-bed`)
     - Pre-built PLINK filesets used in ADMIXTURE tutorials (search
       "1000 genomes hapmap3 plink bed bim fam")
   It's strongly recommended to LD-prune first, outside this script:
       plink --bfile reference --indep-pairwise 50 10 0.1 --out prune
       plink --bfile reference --extract prune.prune.in --make-bed --out reference_pruned
   LD pruning removes redundant, correlated SNPs so PCA reflects population
   structure rather than a few dense LD blocks.

2. A panel file mapping sample ID -> population label, e.g. 1000 Genomes'
   integrated_call_samples.panel:
       sample  pop     super_pop   gender
       HG00096 GBR     EUR         male
       ...

Outputs
-------
model/reference_model.npz containing:
    snp_ids            (n_snps,)   rsIDs, in a fixed order
    ref_allele         (n_snps,)   reference/coding allele per SNP
    alt_allele         (n_snps,)   alternate allele per SNP
    allele_freq        (n_snps,)   reference-panel frequency of ref_allele
    pca_mean           (n_snps,)   mean used to center genotypes before PCA
    pca_components      (k, n_snps) PCA loading matrix (already includes the
                                    1/sqrt(2p(1-p)) scaling baked in, so
                                    projection is a single matrix multiply)
    ref_pcs            (n_samples, k)  reference samples' PCA coordinates
    ref_super_pop      (n_samples,)    label used for classification
    ref_pop            (n_samples,)    finer-grained label, optional display

Dependencies (offline only - NOT deployed):
    pip install pandas-plink numpy pandas scikit-learn
"""

import numpy as np
import pandas as pd
from pandas_plink import read_plink1_bin
from sklearn.decomposition import PCA

# ---- config ----------------------------------------------------------
BED_PREFIX = "reference_pruned"          # expects .bed/.bim/.fam with this prefix
PANEL_FILE = "integrated_call_samples.panel"
N_COMPONENTS = 20
MAF_MIN = 0.05                           # drop very rare/monomorphic SNPs
OUT_PATH = "model/reference_model.npz"

AMBIGUOUS_PAIRS = {frozenset(("A", "T")), frozenset(("C", "G"))}


def is_strand_ambiguous(a0: str, a1: str) -> bool:
    return frozenset((a0, a1)) in AMBIGUOUS_PAIRS


def main():
    print("Loading PLINK fileset...")
    G = read_plink1_bin(f"{BED_PREFIX}.bed", f"{BED_PREFIX}.bim", f"{BED_PREFIX}.fam", verbose=False)
    # G dims: (sample, variant); values in {0,1,2,nan} = ALT allele dosage

    snp_ids = G.snp.values.astype(str)
    a0 = G.a0.values.astype(str)  # reference/major allele in this encoding
    a1 = G.a1.values.astype(str)  # alt allele, dosage counts copies of a1
    sample_ids = G.sample.values.astype(str)

    geno = G.values  # (n_samples, n_snps), float, NaN = missing

    print(f"Loaded {geno.shape[1]} SNPs x {geno.shape[0]} samples")

    # ---- QC ----
    freq = np.nanmean(geno, axis=0) / 2.0  # allele frequency of a1
    missing_rate = np.isnan(geno).mean(axis=0)

    keep = np.ones(len(snp_ids), dtype=bool)
    keep &= missing_rate < 0.02
    keep &= (freq > MAF_MIN) & (freq < 1 - MAF_MIN)
    keep &= np.array([not is_strand_ambiguous(x, y) for x, y in zip(a0, a1)])

    print(f"QC keeps {keep.sum()} / {len(keep)} SNPs")
    geno = geno[:, keep]
    snp_ids = snp_ids[keep]
    a0 = a0[keep]
    a1 = a1[keep]
    freq = freq[keep]

    # mean-impute any remaining missing genotypes with 2*freq
    nan_mask = np.isnan(geno)
    fill = (2 * freq)[np.newaxis, :]
    geno = np.where(nan_mask, np.broadcast_to(fill, geno.shape), geno)

    # ---- standardize (Patterson et al. scaling used by EIGENSOFT/smartpca) ----
    p = freq
    denom = np.sqrt(2 * p * (1 - p))
    denom[denom == 0] = 1.0
    mean = 2 * p
    X = (geno - mean) / denom

    # ---- PCA ----
    print(f"Fitting PCA with {N_COMPONENTS} components...")
    pca = PCA(n_components=N_COMPONENTS, svd_solver="randomized", random_state=0)
    ref_pcs = pca.fit_transform(X)

    # fold the (x-mean)/denom scaling into the components so that, at
    # inference time, projection is: pcs = (raw_dosage - pca_mean) @ pca_components.T
    pca_components = pca.components_ / denom[np.newaxis, :]  # (k, n_snps)
    pca_mean = mean  # raw dosage mean, i.e. 2p

    # ---- attach population labels ----
    panel = pd.read_csv(PANEL_FILE, sep="\t")
    panel = panel.set_index(panel.columns[0])
    super_pop = panel.loc[sample_ids, "super_pop"].values.astype(str)
    pop = panel.loc[sample_ids, "pop"].values.astype(str)

    print("Saving model artifacts...")
    np.savez_compressed(
        OUT_PATH,
        snp_ids=snp_ids,
        ref_allele=a0,
        alt_allele=a1,
        allele_freq=freq,
        pca_mean=pca_mean,
        pca_components=pca_components,
        ref_pcs=ref_pcs,
        ref_super_pop=super_pop,
        ref_pop=pop,
    )
    print(f"Wrote {OUT_PATH}  ({ref_pcs.shape[0]} samples, {len(snp_ids)} SNPs, {N_COMPONENTS} PCs)")


if __name__ == "__main__":
    main()
