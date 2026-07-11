"""Export static frontend assets from api/model/reference_model.npz.

Two files go into public/:
  reference_snps.json     -- list of rsIDs the model uses (lets the
                              frontend pre-filter a huge raw export client-
                              side before uploading it)
  reference_scatter.json  -- PC1/PC2 for every reference sample, with BOTH
                              continental (super_pop) and detailed (pop)
                              labels, so the frontend can draw the
                              background cloud AND compute/highlight
                              detailed-population centroids (e.g. "GBR",
                              "PUR") without any extra backend work.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "api", "model", "reference_model.npz")
PUBLIC = os.path.join(ROOT, "public")


def main():
    with np.load(MODEL, allow_pickle=False) as m:
        snps = m["snp_ids"].tolist()
        scatter = {
            "points": m["ref_pcs"][:, :2].tolist(),
            "labels": m["ref_super_pop"].tolist(),      # continental, e.g. "EUR"
            "pop_labels": m["ref_pop"].tolist(),         # detailed, e.g. "GBR"
        }
    os.makedirs(PUBLIC, exist_ok=True)
    snp_path = os.path.join(PUBLIC, "reference_snps.json")
    scatter_path = os.path.join(PUBLIC, "reference_scatter.json")
    with open(snp_path, "w", encoding="utf-8") as f:
        json.dump(snps, f)
    with open(scatter_path, "w", encoding="utf-8") as f:
        json.dump(scatter, f)
    print(f"Wrote {snp_path} ({len(snps)} SNPs)")
    print(f"Wrote {scatter_path} ({len(scatter['points'])} points)")


if __name__ == "__main__":
    sys.exit(main())
