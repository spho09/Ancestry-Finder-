# Genetic ancestry estimator

Estimates genetic-ancestry proportions (relative to 1000 Genomes reference
populations) from an uploaded SNP file, via PCA + k-NN. Deployed as a static
frontend + one lightweight Python serverless function on Vercel.

## Why the pipeline is split into "offline" and "online"

Vercel functions are ephemeral and have tight size/time/memory limits. You
cannot load a full reference panel (hundreds of thousands of SNPs x
thousands of samples), run QC, and fit PCA inside a single request. So:

- **`offline/`** — run locally, once (or whenever you refresh the reference
  panel). Loads the raw reference data, does QC, fits PCA, saves a small
  (`.npz`) file with just the PCA loadings + reference samples' coordinates.
- **`api/predict.py`** — the actual deployed function. Loads that small
  `.npz` file, and for each uploaded sample does a single matrix
  multiplication to project into PCA space, then a k-NN lookup. Fast, and
  only depends on `numpy`.

## 1. Get a reference panel

Download 1000 Genomes Phase 3 (or a pre-built PLINK fileset commonly used
in ADMIXTURE/PCA tutorials — search "1000 genomes hapmap3 plink bed bim
fam") plus the panel file mapping sample -> population:
`integrated_call_samples.panel`, available from
https://www.internationalgenome.org/data

LD-prune before running the build script (removes redundant correlated
SNPs so PCA reflects population structure, not dense LD blocks):

```bash
plink --bfile reference --indep-pairwise 50 10 0.1 --out prune
plink --bfile reference --extract prune.prune.in --make-bed --out reference_pruned
```

## 2. Build the model artifacts (offline, local machine)

```bash
cd offline
pip install -r requirements.txt
python build_reference_model.py
```

This writes `offline/model/reference_model.npz`. Copy it into
`api/model/reference_model.npz` before deploying:

```bash
mkdir -p ../api/model
cp model/reference_model.npz ../api/model/
```

## 3. Deploy to Vercel

```bash
npm i -g vercel   # if you don't have it
vercel
```

Vercel will pick up `vercel.json`, install `api/requirements.txt` (just
`numpy`), and serve `public/index.html` as the static site.

## 4. Supported upload formats

- 23andMe / AncestryDNA raw text export (`rsid  chromosome  position  genotype`)
- Single-sample VCF (biallelic SNPs; indels/multiallelic sites are skipped)

Both must use the **same genome build** as your reference panel (check
this — build mismatches are the most common cause of low SNP overlap).

## Accuracy notes

- k-NN-in-PCA-space proportions are a reasonable, deployable proxy but are
  **not** the same as a true ADMIXTURE run — ADMIXTURE is a separate,
  compiled tool that models admixture directly and would need to run
  offline (e.g. in supervised/projection mode against the reference) if
  you want proportions closer to the ADMIXTURE gold standard.
- Results describe genetic similarity to reference populations sampled by
  1000 Genomes, not self-identified ethnicity — those are different things,
  and the app's own output note says so.
- Nothing is persisted server-side in `api/predict.py` as written; if you
  add storage/logging of uploaded genetic data, that's sensitive personal
  data in most jurisdictions (GDPR "special category" data, similar rules
  elsewhere) — handle consent/retention accordingly.

## Possible extensions

- Send reference samples' PC1/PC2 down to the frontend (a few thousand
  points, small payload) to draw the user's point over the actual
  reference population clusters instead of a bare axis.
- Add finer-grained population labels (not just super-population) once
  you've validated k-NN performs well at that resolution.
- Swap the manual VCF/23andMe parser for `cyvcf2`/`hail` if you need to
  support more input formats — trade-off is a heavier function bundle.
