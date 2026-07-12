# Ancestry Finder — setup checklist

Get the app from "deployed but empty" to actually estimating ancestry from uploaded SNP files.

Each task has a **Done when** line — that's how you know it worked.

---

## Current status (your machine)

| Item | Status |
|------|--------|
| Frontend + API code | Done |
| Vercel deploy | Done — https://ancestry-finder-sooty.vercel.app |
| `1kg/` reference download | **Done — correct fileset** (see section 2) |
| `integrated_call_samples.panel` | **Done** — `offline/integrated_call_samples.panel` (2505 lines, 54 KB) |
| Merged PLINK file in `offline/` | **Not done yet** (files are still split by chromosome) |
| `api/model/reference_model.npz` | **Done** — 6.8 MB (80,975 SNPs, 2504 samples) |
| Merged + pruned reference in `offline/` | **Done** |
| Production redeploy | **Waiting on you** — run `vercel --prod` |

---

## 1. Install tools

### PLINK
- [ ] Run: `brew install plink`
- **Purpose:** Merge per-chromosome files into one fileset; optionally LD-prune SNPs.
- **Done when:** `plink --version` prints a version (e.g. `PLINK v1.9` or `v2.0`).

### Python offline environment
- [ ] Run:
  ```bash
  cd offline
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- **Purpose:** Run `build_reference_model.py` locally (not on Vercel).
- **Done when:** All of these import without error:
  ```bash
  python -c "import numpy, pandas, sklearn; from pandas_plink import read_plink1_bin; print('ok')"
  ```
  Prints `ok`.

### Vercel CLI
- [x] Installed and logged in
- **Done when:** `vercel --version` works (already true for you).

---

## 2. Reference panel download — **YOU ARE HERE (mostly done)**

You downloaded `1kg/`. **Yes, this is the right thing.**

### How to recognize the correct fileset

Your files should look like this:

```
1kg/
  1KGPhase3.w_hm3.chr1.bed
  1KGPhase3.w_hm3.chr1.bim
  1KGPhase3.w_hm3.chr1.fam
  1KGPhase3.w_hm3.chr2.bed
  ...
  1KGPhase3.w_hm3.chr22.bed
  ...
```

**What the name means:**
- `1KGPhase3` = 1000 Genomes Phase 3
- `w_hm3` = **with HapMap3 SNPs** — the standard curated SNP list used in PCA/admixture tutorials. This *is* the "HapMap3 subset" the README refers to.

**Done when (checklist):**
- [x] Folder contains `.bed`, `.bim`, `.fam` triplets
- [x] Filenames include `1KGPhase3` and `w_hm3`
- [x] Chromosomes 1–22 present (22 `.bed` files) — yours: **22 chromosomes**
- [x] ~2,500 samples — yours: **2,504** (check: `wc -l 1kg/1KGPhase3.w_hm3.chr1.fam`)
- [x] ~1.1–1.4 million SNPs total — yours: **1,330,820** across all chromosomes
- [x] Sample IDs look like `HG00096`, `NA19238` — yours: **yes**

**Quick self-check command:**
```bash
wc -l 1kg/1KGPhase3.w_hm3.chr1.fam    # expect ~2504
head -1 1kg/1KGPhase3.w_hm3.chr1.fam  # expect HG00096 HG00096 ...
head -1 1kg/1KGPhase3.w_hm3.chr1.bim  # expect: 1  rs...  0  position  A  G
```

You do **not** need to download anything else for the genotype data.

---

## 3. Population panel file — **DONE**

- [x] Downloaded to `offline/integrated_call_samples.panel`

**Purpose:** Maps each sample ID (e.g. `HG00096`) to population labels (`GBR`, super-pop `EUR`).

**Done when:**
- File exists at `offline/integrated_call_samples.panel`
- First line is a header like: `sample  pop  super_pop  gender`
- `wc -l offline/integrated_call_samples.panel` → **~3505 lines** (header + ~3504 samples)
- This command finds your reference samples:
  ```bash
  head -1 offline/integrated_call_samples.panel
  grep -c '^HG' offline/integrated_call_samples.panel   # expect hundreds of matches
  ```

---

## 4. Merge chromosomes into one PLINK fileset

The build script expects **one** merged fileset in `offline/`, not 22 separate chromosome files.

- [ ] Run from project root:
  ```bash
  cd offline

  # list all chromosome prefixes for PLINK merge
  ls ../1kg/1KGPhase3.w_hm3.chr*.bed \
    | sed 's/.bed$//' \
    | sort -V \
    > merge_list.txt

  plink --merge-list merge_list.txt --make-bed --out reference
  ```

**Purpose:** Combine chr1–chr22 into a single `reference.bed/.bim/.fam` that Python can load at once.

**Done when:**
- These three files exist in `offline/`:
  - `reference.bed` (large, ~800MB+)
  - `reference.bim` (~1.3M lines)
  - `reference.fam` (2504 lines)
- PLINK ends with something like `2504 people loaded` and no fatal errors
- Verify:
  ```bash
  wc -l reference.bim reference.fam
  # expect: ~1330820 reference.bim
  # expect:     2504 reference.fam
  ```

**If merge fails** with "multiallelic" or duplicate SNP errors, say so — there are fix flags, but try the plain merge first.

---

## 5. LD-prune (recommended, then rename)

Your data is already restricted to HapMap3 SNPs, but the README still recommends pruning correlated SNPs within that set.

- [ ] Run:
  ```bash
  cd offline
  plink --bfile reference --indep-pairwise 50 10 0.1 --out prune
  plink --bfile reference --extract prune.prune.in --make-bed --out reference_pruned
  ```

**Purpose:** Drop SNPs that are redundant with nearby SNPs so PCA captures population structure, not local LD blocks.

**Done when:**
- These three files exist in `offline/`:
  - `reference_pruned.bed`
  - `reference_pruned.bim`
  - `reference_pruned.fam`
- SNP count drops from ~1.33M to something smaller (often **~100k–300k** — exact number varies):
  ```bash
  wc -l reference_pruned.bim   # expect well under 1,330,820
  wc -l reference_pruned.fam   # still 2504
  ```

`build_reference_model.py` is hardcoded to read the prefix `reference_pruned` from the `offline/` directory — **this exact name matters**.

---

## 6. Build the model

- [ ] Run:
  ```bash
  cd offline
  source .venv/bin/activate
  python build_reference_model.py
  ```

**What it does:** QC → standardize genotypes → fit 20-component PCA → attach population labels → save compressed model.

**Done when terminal output looks roughly like:**
```
Loading PLINK fileset...
Loaded 1330820 SNPs x 2504 samples        # or similar starting count
QC keeps XXXXX / XXXXX SNPs               # expect tens of thousands kept (not 0)
Fitting PCA with 20 components...
Saving model artifacts...
Wrote model/reference_model.npz  (2504 samples, XXXXX SNPs, 20 PCs)
```

**Done when files look like:**
- `offline/model/reference_model.npz` exists
- File size is **tens of MB** (e.g. 20–80 MB), **not** 2 bytes
  ```bash
  ls -lh offline/model/reference_model.npz
  ```

**Failed if:**
- `FileNotFoundError` for `reference_pruned.bed` → section 4 or 5 not done
- `KeyError` on panel → section 3 not done
- `QC keeps 0 / ... SNPs` → input data problem; check merge output

---

## 7. Copy model and redeploy

- [ ] Run:
  ```bash
  cp offline/model/reference_model.npz api/model/reference_model.npz
  ls -lh api/model/reference_model.npz    # same large size as offline copy
  cd ..
  vercel --prod
  ```

**Done when:**
- `api/model/reference_model.npz` matches offline copy in size (not 2 bytes)
- `vercel --prod` finishes with a production URL and no errors

---

## 8. Test end-to-end

- [ ] Upload a **23andMe / AncestryDNA raw export** (`.txt`) or **single-sample VCF** at https://ancestry-finder-sooty.vercel.app

**Done when the site shows:**
- Ancestry proportion bars (AFR / AMR / EAS / EUR / SAS)
- Closest population + confidence %
- **SNPs used: ≥ 500** (ideally tens of thousands)
- A dot on the PCA plot
- No red error message

**Failed if you see:**
| Error | Likely cause |
|-------|----------------|
| `Only parsed N genotype calls` | Wrong file format, or file is empty/corrupt |
| `Only N SNPs overlap the reference panel` | Genome build mismatch (hg19 vs hg38). 1000 Genomes Phase 3 is **GRCh37/hg19**. Use a raw export on the same build. |
| `Internal error` | Model file still placeholder, or deploy didn't include `api/model/` |

---

## Folder layout when everything is done

```
Ancestry-Finder-/
├── 1kg/                              # raw per-chromosome download (keep as archive)
│   └── 1KGPhase3.w_hm3.chr{1..22}.*
├── offline/
│   ├── integrated_call_samples.panel # population labels
│   ├── reference.bed/.bim/.fam       # merged (intermediate)
│   ├── reference_pruned.bed/.bim/.fam  # what build script reads
│   ├── model/reference_model.npz     # built model (~tens of MB)
│   └── .venv/
├── api/
│   └── model/reference_model.npz     # copy of offline model (deployed)
└── public/index.html                 # frontend
```

---

## Optional follow-ups

- [ ] Connect GitHub in Vercel (failed first time — needs GitHub login connection in Vercel account settings)
- [ ] Add reference PC scatter to frontend
- [ ] Finer-grained population labels beyond super-populations

---

## Future improvements (not required now, left as TODOs in code)

- **Larger reference panel**: 1000 Genomes (2,504 samples) is a good starting
  point but under-represents many world populations. HGDP (Human Genome
  Diversity Project, ~1,000 samples, more geographically diverse) or a
  1000G + HGDP merge would improve coverage. Would require re-running
  `offline/build_reference_model.py` against the merged panel.
- **ADMIXTURE-style modeling**: the current inference (RBF/Gaussian
  similarity to PCA centroids) is a fast proxy for population membership.
  A true ADMIXTURE run (explicit generative model of allele frequencies
  under k-way admixture) would give more rigorous ancestry proportions,
  at the cost of needing to run offline (ADMIXTURE is a separate compiled
  tool, not something to run per-request in a serverless function).
- **Full covariance instead of scalar spread**: `pop_spread` is currently
  one number per population (isotropic). A full covariance matrix per
  population (Mahalanobis distance) would better capture populations
  whose genetic variation isn't equally spread in every PCA direction —
  1000 Genomes' smaller populations (60-100 samples) are borderline for
  estimating a stable 20x20 covariance matrix, which is why this version
  uses a scalar.
- **UMAP visualization**: PCA is used for both inference and the on-page
  scatter plot. A separate UMAP embedding (fit only for visualization,
  not inference) can show cluster structure more clearly than PC1/PC2
  alone, since it isn't limited to linear projections.
- **Confidence intervals / bootstrap uncertainty**: current confidence is
  a single entropy-based score. Bootstrapping over random SNP subsets
  (re-run projection+scoring on e.g. 100 resamples of the SNP set) would
  give an actual uncertainty range around each population's probability,
  rather than one point estimate.

---

## Hybrid PCA + allele-frequency pipeline (this update)

**What changed:** inference is now two-stage. PCA still projects and does a
fast first-pass similarity ranking (unchanged from before), but its output
is now treated as a *prior* over candidate populations (top 10 of 26),
which gets refined by a Hardy-Weinberg genotype-likelihood score computed
from precomputed per-population allele frequencies (new offline artifact:
`pop_allele_freq`, (26, n_snps)). The two signals are combined via
Bayes' rule in log-space and softmax-normalized. See `api/predict.py`'s
module docstring for the full writeup.

**Conceptually borrowed from AEON** (github.com/[aeon project], not
copied): modeling ancestry via allele frequencies and Hardy-Weinberg
genotype probabilities, rather than PCA distance alone. AEON fits a
continuous admixture-proportion vector via gradient-based MLE (Pyro/
PyTorch, SVI/MCMC) -- this project instead scores a small, PCA-narrowed
set of discrete candidate populations once, via closed-form vectorized
NumPy, with no optimizer and no new dependency, to stay within a
serverless time/memory budget.

**Validation:** re-ran the 10-sample known-ancestry test suite (see
`test_data/`). Continent-level accuracy: 9/10 (unchanged). Detailed-
population accuracy: improved from 8/10 to 9/10. Runtime: ~150-200ms
per request warm, ~350ms cold -- well within the 3s target.

**Important limitation on this validation:** the 10 test samples are
individuals who WERE part of the 2,504-sample reference panel used to
build the model (their genotypes contributed to the population centroids,
spreads, and allele frequencies they're being compared against). This is
in-sample accuracy, not held-out/generalization accuracy, and will be
somewhat optimistic relative to how the model performs on a genuinely new
genome. A rigorous validation would re-run `build_reference_model.py`
with each test sample's population excluded (leave-one-out), which is a
larger undertaking than in-sample testing and hasn't been done here.
TODO(future work): implement leave-one-out validation for a proper
generalization estimate.

**On the "American" bias:** the AMR populations (MXL/PUR/CLM/PEL) have
the widest PCA spread of any group in the panel because they're
themselves admixed -- this made PCA-only inference prone to defaulting
ambiguous or out-of-panel genomes toward AMR, since a wide cluster
accepts a wider range of points. The allele-frequency likelihood stage is
a different signal (genotype fit, not projection distance) and isn't
subject to the same bias, so it can and does pull probability away from
AMR when the genotype evidence doesn't support it. This can't be fully
validated against genomes truly outside 1000 Genomes' coverage (Central
Asian, Siberian, Middle Eastern, North African) since no such samples
exist in this reference panel to test against -- that's a structural
limitation of the reference panel itself, not something the inference
algorithm can fix. Expanding the reference panel (see earlier TODO on
HGDP integration) is the real fix for that gap.
