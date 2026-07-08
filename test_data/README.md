# Test SNP files

Synthetic **23andMe-style** exports generated from public 1000 Genomes Phase 3 samples (hg19). Safe to use for testing — not real personal data.

## Files

| File | Population | Expected super-pop |
|------|------------|-------------------|
| `HG00096_23andme.txt` | British (GBR) | **EUR** |
| `HG01537_23andme.txt` | Iberian Spanish (IBS) | **EUR** |
| `NA19238_23andme.txt` | Yoruba (YRI) | **AFR** |
| `NA19017_23andme.txt` | Luhya Kenyan (LWK) | **AFR** |
| `HG00419_23andme.txt` | Southern Han (CHS) | **EAS** |
| `NA18939_23andme.txt` | Japanese (JPT) | **EAS** |
| `HG01565_23andme.txt` | Peruvian (PEL) | **AMR** |
| `NA19648_23andme.txt` | Mexican (MXL) | **AMR** |
| `HG03805_23andme.txt` | Bengali (BEB) | **SAS** |
| `HG01583_23andme.txt` | Punjabi (PJL) | **SAS** |

Each file has ~134k SNPs and should overlap ~80k+ SNPs with the deployed reference model.

## How to test

1. Open your deployed site
2. Upload one of the `.txt` files
3. Click **Estimate ancestry**
4. Check closest population and PCA cluster match the table above

## Regenerate one file

```bash
cd offline
ID=HG00096
echo "$ID $ID" > ../test_data/$ID.keep
../tools/plink --bfile reference_pruned --keep ../test_data/$ID.keep \
  --recode 23 --out ../test_data/${ID}_23andme
# PLINK may write .23 or .txt depending on version — rename if needed:
test -f ../test_data/${ID}_23andme.23 && mv ../test_data/${ID}_23andme.23 ../test_data/${ID}_23andme.txt
rm -f ../test_data/$ID.keep ../test_data/${ID}_23andme.log ../test_data/${ID}_23andme.nosex
```
