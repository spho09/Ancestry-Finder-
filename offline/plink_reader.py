import numpy as np
import pandas as pd

BIM_COLS = ["chrom", "snp_id", "cm", "pos", "a1", "a2"]
FAM_COLS = ["fid", "iid", "pid", "mid", "sex", "phenotype"]
CODE_TO_DOSAGE = np.array([0.0, np.nan, 1.0, 2.0], dtype=np.float32)


def read_bim(prefix: str) -> pd.DataFrame:
    return pd.read_csv(f"{prefix}.bim", sep=r"\s+", header=None, names=BIM_COLS)


def read_fam(prefix: str) -> pd.DataFrame:
    return pd.read_csv(f"{prefix}.fam", sep=r"\s+", header=None, names=FAM_COLS)


def read_bed(prefix: str, n_samples: int, n_snps: int) -> np.ndarray:
    with open(f"{prefix}.bed", "rb") as f:
        magic = f.read(3)
        if magic != b"\x6c\x1b\x01":
            raise ValueError(f"{prefix}.bed: not a SNP-major PLINK 1 bed file")
        raw = np.frombuffer(f.read(), dtype=np.uint8)

    bytes_per_snp = (n_samples + 3) // 4
    expected = bytes_per_snp * n_snps
    if raw.size != expected:
        raise ValueError(
            f"{prefix}.bed: expected {expected} bytes for {n_snps} SNPs x "
            f"{n_samples} samples, got {raw.size}"
        )

    raw = raw.reshape(n_snps, bytes_per_snp)
    bits = np.unpackbits(raw, axis=1, bitorder="little")
    bits = bits.reshape(n_snps, bytes_per_snp * 4, 2)
    codes = bits[:, :, 0] + 2 * bits[:, :, 1]
    codes = codes[:, :n_samples]

    dosage = CODE_TO_DOSAGE[codes]
    return dosage.T


def load_plink(prefix: str):
    fam = read_fam(prefix)
    bim = read_bim(prefix)
    geno = read_bed(prefix, n_samples=len(fam), n_snps=len(bim))
    return geno, bim, fam
