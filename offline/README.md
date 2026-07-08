# Offline build (local only — not deployed)

Run once on your machine to build `model/reference_model.npz`, then copy to `api/model/`.

## What to keep

| Path | Purpose | Size |
|------|---------|------|
| `integrated_call_samples.panel` | Sample → population labels | ~54 KB |
| `reference_pruned.bed/.bim/.fam` | LD-pruned reference for PCA | ~84 MB |
| `model/reference_model.npz` | Built model (copy to `api/model/`) | ~7 MB |
| `build_reference_model.py` | Build script | — |
| `.venv/` | Python deps | — |

## Raw source data (project root)

| Path | Purpose |
|------|---------|
| `../1kg/` | Per-chromosome 1000 Genomes download (only needed to re-merge/re-prune) |
| `../tools/plink` | PLINK binary for merge/prune/export |

## Rebuild from scratch

If you deleted `reference_pruned.*`, regenerate from `1kg/`:

```bash
cd offline
ls ../1kg/1KGPhase3.w_hm3.chr*.bed | sed 's/.bed$//' | sort -V > merge_list.txt
../tools/plink --merge-list merge_list.txt --make-bed --out reference
../tools/plink --bfile reference --indep-pairwise 50 10 0.1 --out prune
../tools/plink --bfile reference --extract prune.prune.in --make-bed --out reference_pruned
rm -f reference.bed reference.bim reference.fam prune.* reference.log reference.nosex merge_list.txt
source .venv/bin/activate
python build_reference_model.py
cp model/reference_model.npz ../api/model/
```

## Clean temp files after merge/prune

```bash
./cleanup.sh
```

Removes PLINK logs, prune lists, and the large unpruned `reference.*` merge (safe once `reference_pruned.*` exists).
