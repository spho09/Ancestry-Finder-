# Large test files (~32 MB each)

Full **23andMe-style** exports from the complete 1000 Genomes HapMap3 panel (**~1.33 million SNPs** per file). These simulate real consumer raw downloads that are much bigger than the small `test_data/*_23andme.txt` files.

| File | Size | SNP lines | Expected result |
|------|------|-----------|-----------------|
| `HG00096_full_23andme.txt` | ~32 MB | ~1,330,820 | **EUR** |
| `NA19238_full_23andme.txt` | ~32 MB | ~1,330,820 | **AFR** |
| `HG00419_full_23andme.txt` | ~32 MB | ~1,330,820 | **EAS** |

## Why these exist

Real 23andMe/AncestryDNA exports are **~15–50 MB** with **~600k+ SNPs**. The app filters client-side to ~81k reference SNPs before upload. These files stress-test that path.

## Regenerate

```bash
./scripts/build_large_test_files.sh
```

Requires `1kg/` reference data and `tools/plink`.
