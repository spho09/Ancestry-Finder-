# Test SNP files

Synthetic **23andMe-style** exports generated from public 1000 Genomes Phase 3 samples (hg19). Safe to use for testing — not real personal data.

## Files

| File | Sample | Expected ancestry |
|------|--------|-------------------|
| `HG00096_23andme.txt` | British (GBR) | Mostly **EUR** (European) |
| `NA19238_23andme.txt` | Yoruba (YRI) | Mostly **AFR** (African) |
| `HG00419_23andme.txt` | Southern Han (CHS) | Mostly **EAS** (East Asian) |

Each file has ~134k SNPs and should overlap ~80k+ SNPs with the deployed reference model.

## How to test

1. Open https://ancestry-finder-sooty.vercel.app (after `vercel --prod` with the real model)
2. Upload one of the `.txt` files
3. Click **Estimate ancestry**
4. Check that closest population matches the table above

## Regenerate

```bash
cd offline
# example for HG00096
echo "HG00096 HG00096" > ../test_data/HG00096.keep
../tools/plink --bfile reference_pruned --keep ../test_data/HG00096.keep \
  --recode 23 --out ../test_data/HG00096_23andme
mv ../test_data/HG00096_23andme.23 ../test_data/HG00096_23andme.txt
rm -f ../test_data/HG00096.keep ../test_data/HG00096_23andme.log ../test_data/HG00096_23andme.nosex
```

Only the `*_23andme.txt` files are needed for upload testing.
