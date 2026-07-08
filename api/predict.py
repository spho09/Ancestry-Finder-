"""
api/predict.py

Vercel Python serverless function. Loads the small precomputed reference
model (built offline, see offline/build_reference_model.py) and, per
request:
  1. parses an uploaded SNP file (23andMe/AncestryDNA raw text, or VCF)
  2. intersects it against the reference SNP set
  3. QCs + encodes + standardizes genotypes
  4. projects into the reference PCA space
  5. estimates ancestry proportions via distance-weighted k-NN among
     reference samples in PCA space
  6. returns JSON with proportions, closest population, PCA coordinates,
     and a confidence/QC summary

Only depends on numpy at request time -- keeps the deployed bundle small
and cold starts fast.
"""
import traceback
import json
import os
from http.server import BaseHTTPRequestHandler

import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "reference_model.npz")
K_NEIGHBORS = 30
MIN_SNPS_REQUIRED = 500  # below this, results are too noisy to report

_model = None  # lazy-loaded, cached across warm invocations


def load_model():
    global _model
    if _model is None:
        with np.load(MODEL_PATH, allow_pickle=False) as data:
            _model = {k: data[k] for k in data.files}
    return _model


def parse_23andme_raw(text: str):
    """rsid\tchromosome\tposition\tgenotype, '#' comment lines."""
    calls = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.strip().split("\t")
        if len(parts) < 4:
            parts = line.strip().split(",")
        if len(parts) < 4:
            continue
        rsid, _chrom, _pos, genotype = parts[0], parts[1], parts[2], parts[3]
        genotype = genotype.replace("-", "")
        if len(genotype) == 2:
            calls[rsid] = (genotype[0], genotype[1])
    return calls


def parse_vcf(text: str):
    """Minimal single-sample VCF parser: rsID column (ID field) -> genotype."""
    calls = {}
    for line in text.splitlines():
        if not line or line.startswith("##"):
            continue
        if line.startswith("#CHROM"):
            continue
        fields = line.strip().split("\t")
        if len(fields) < 10:
            continue
        rsid = fields[2]
        ref, alt = fields[3], fields[4]
        if rsid == "." or len(ref) != 1 or len(alt) != 1:
            continue  # skip indels/unnamed variants for simplicity
        fmt = fields[8].split(":")
        sample = fields[9].split(":")
        try:
            gt = sample[fmt.index("GT")]
        except ValueError:
            continue
        alleles = gt.replace("|", "/").split("/")
        if len(alleles) != 2 or "." in alleles:
            continue
        base = {"0": ref, "1": alt}
        a1 = base.get(alleles[0])
        a2 = base.get(alleles[1])
        if a1 and a2:
            calls[rsid] = (a1, a2)
    return calls


def detect_and_parse(text: str):
    if text.lstrip().startswith("##fileformat=VCF") or "\n#CHROM" in text:
        return parse_vcf(text)
    return parse_23andme_raw(text)


def build_feature_vector(calls: dict, model: dict):
    snp_ids = model["snp_ids"]
    ref_allele = model["ref_allele"]
    alt_allele = model["alt_allele"]
    allele_freq = model["allele_freq"]  # freq of alt_allele in reference
    pca_mean = model["pca_mean"]

    n = len(snp_ids)
    dosage = 2 * allele_freq.copy()  # default: mean-impute missing SNPs
    used_mask = np.zeros(n, dtype=bool)

    # index lookup for the (much smaller) set of overlapping SNPs
    id_to_idx = {rsid: i for i, rsid in enumerate(snp_ids)}
    n_matched = 0
    n_ambiguous_dropped = 0

    for rsid, (a1, a2) in calls.items():
        idx = id_to_idx.get(rsid)
        if idx is None:
            continue
        ref = ref_allele[idx]
        alt = alt_allele[idx]
        obs = {a1, a2}
        allowed = {ref, alt}
        if not obs.issubset(allowed):
            # allele mismatch (build mismatch, genotyping error, etc.) - skip
            n_ambiguous_dropped += 1
            continue
        count_alt = (a1 == alt) + (a2 == alt)
        dosage[idx] = count_alt
        used_mask[idx] = True
        n_matched += 1

    return dosage, used_mask, n_matched, n_ambiguous_dropped


def project_and_classify(dosage: np.ndarray, model: dict):
    pca_mean = model["pca_mean"]
    pca_components = model["pca_components"]  # (k, n_snps)
    ref_pcs = model["ref_pcs"]                 # (n_ref, k)
    ref_super_pop = model["ref_super_pop"]

    user_pcs = (dosage - pca_mean) @ pca_components.T  # (k,)

    dists = np.linalg.norm(ref_pcs - user_pcs[np.newaxis, :], axis=1)
    nn_idx = np.argsort(dists)[:K_NEIGHBORS]
    nn_dists = dists[nn_idx]
    nn_labels = ref_super_pop[nn_idx]

    weights = 1.0 / (nn_dists + 1e-6)
    weights /= weights.sum()

    proportions = {}
    for label, w in zip(nn_labels, weights):
        proportions[label] = proportions.get(label, 0.0) + float(w)

    closest_population = max(proportions, key=proportions.get)
    # simple confidence heuristic: how dominant is the top population,
    # and how tight is the nearest-neighbor cloud
    top_share = proportions[closest_population]

    return {
        "pca_coordinates": user_pcs[:2].tolist(),  # PC1/PC2 for plotting
        "ancestry_proportions": {k: round(v, 4) for k, v in proportions.items()},
        "closest_population": closest_population,
        "confidence": round(float(top_share), 3),
        "mean_neighbor_distance": round(float(nn_dists.mean()), 3),
    }


def handle_request(body: bytes) -> dict:
    model = load_model()
    text = body.decode("utf-8", errors="ignore")
    calls = detect_and_parse(text)

    if len(calls) < MIN_SNPS_REQUIRED:
        return {
            "error": (
                f"Only parsed {len(calls)} genotype calls from the uploaded file. "
                f"Need at least {MIN_SNPS_REQUIRED} to produce a reliable estimate. "
                "Check that the file is a 23andMe/AncestryDNA raw export or a "
                "single-sample VCF."
            )
        }

    dosage, used_mask, n_matched, n_dropped = build_feature_vector(calls, model)

    if n_matched < MIN_SNPS_REQUIRED:
        return {
            "error": (
                f"Only {n_matched} of your SNPs overlap the reference panel "
                f"(need at least {MIN_SNPS_REQUIRED}). This can happen if your "
                "file uses a different genome build than the reference panel."
            )
        }

    result = project_and_classify(dosage, model)
    result["n_snps_used"] = int(n_matched)
    result["n_snps_total_in_reference"] = int(len(model["snp_ids"]))
    result["n_allele_mismatches_dropped"] = int(n_dropped)
    result["note"] = (
        "These are estimates of genetic ancestry relative to 1000 Genomes "
        "reference populations, not a determination of ethnicity, which is a "
        "social and cultural category that does not map directly onto genetic "
        "clusters."
    )
    return result


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            result = handle_request(body)
            status = 400 if "error" in result else 200
            self._respond(status, result)
        except Exception:
    error = traceback.format_exc()
    print(error)

    self._respond(
        500,
        {
            "error": "Internal Server Error",
            "traceback": error
        }
    )

    def do_OPTIONS(self):
        self._respond(204, None)

    def _respond(self, status: int, payload):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        if payload is not None:
            self.wfile.write(json.dumps(payload).encode("utf-8"))
