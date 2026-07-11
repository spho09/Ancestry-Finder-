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
        snp_ids = _model["snp_ids"]
        _model["id_to_idx"] = {str(rsid): i for i, rsid in enumerate(snp_ids)}
        _model["needed_rsids"] = set(_model["id_to_idx"])
    return _model


def _parse_23andme_line(line: str, id_to_idx: dict, calls: dict) -> None:
    parts = line.split("\t")
    if len(parts) < 4:
        parts = line.split(",")
    if len(parts) < 4:
        return
    rsid = parts[0]
    idx = id_to_idx.get(rsid)
    if idx is None:
        return
    genotype = parts[3].replace("-", "")
    if len(genotype) == 2:
        calls[rsid] = (genotype[0], genotype[1])


def _parse_vcf_line(line: str, id_to_idx: dict, calls: dict) -> None:
    fields = line.split("\t")
    if len(fields) < 10:
        return
    rsid = fields[2]
    idx = id_to_idx.get(rsid)
    if idx is None:
        return
    ref, alt = fields[3], fields[4]
    if rsid == "." or len(ref) != 1 or len(alt) != 1:
        return
    fmt = fields[8].split(":")
    sample = fields[9].split(":")
    try:
        gt = sample[fmt.index("GT")]
    except ValueError:
        return
    alleles = gt.replace("|", "/").split("/")
    if len(alleles) != 2 or "." in alleles:
        return
    base = {"0": ref, "1": alt}
    a1 = base.get(alleles[0])
    a2 = base.get(alleles[1])
    if a1 and a2:
        calls[rsid] = (a1, a2)


def parse_upload(body: bytes, model: dict) -> dict:
    """Parse only SNPs present in the reference panel (single pass, low memory)."""
    id_to_idx = model["id_to_idx"]
    calls = {}
    is_vcf = False
    header_checked = False

    for raw_line in body.splitlines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        if not header_checked:
            is_vcf = line.startswith("##fileformat=VCF") or line.startswith("#CHROM")
            header_checked = True
        if line.startswith("#"):
            continue
        if is_vcf:
            _parse_vcf_line(line, id_to_idx, calls)
        else:
            _parse_23andme_line(line, id_to_idx, calls)
        # early exit once every reference SNP is found
        if len(calls) >= len(id_to_idx):
            break
    return calls


def build_feature_vector(calls: dict, model: dict):
    snp_ids = model["snp_ids"]
    ref_allele = model["ref_allele"]
    alt_allele = model["alt_allele"]
    allele_freq = model["allele_freq"]

    n = len(snp_ids)
    dosage = 2 * allele_freq.copy()
    id_to_idx = model["id_to_idx"]
    n_matched = 0
    n_ambiguous_dropped = 0

    for rsid, (a1, a2) in calls.items():
        idx = id_to_idx[rsid]
        ref = ref_allele[idx]
        alt = alt_allele[idx]
        if not {a1, a2}.issubset({ref, alt}):
            n_ambiguous_dropped += 1
            continue
        dosage[idx] = (a1 == alt) + (a2 == alt)
        n_matched += 1

    return dosage, n_matched, n_ambiguous_dropped


def project_and_classify(dosage: np.ndarray, model: dict):
    pca_mean = model["pca_mean"]
    pca_components = model["pca_components"]
    ref_pcs = model["ref_pcs"]
    ref_super_pop = model["ref_super_pop"]
    ref_pop = model["ref_pop"]

    # Project uploaded sample into PCA space
    user_pcs = (dosage - pca_mean) @ pca_components.T

    # ----------------------------------------------------
    # Build a centroid for every reference population
    # ----------------------------------------------------
    population_centroids = {}

    for pop in np.unique(ref_pop):
        mask = ref_pop == pop
        population_centroids[pop] = ref_pcs[mask].mean(axis=0)

    # ----------------------------------------------------
    # Distance from user to each population centroid
    # ----------------------------------------------------
    population_scores = {}

    for pop, centroid in population_centroids.items():
        dist = np.linalg.norm(user_pcs - centroid)

        # Similarity instead of raw distance
        score = np.exp(-dist)

        population_scores[pop] = score

    # Normalize to probabilities
    total = sum(population_scores.values())

    if total == 0:
        total = 1.0

    for pop in population_scores:
        population_scores[pop] /= total

    # ----------------------------------------------------
    # Aggregate into continental groups
    # ----------------------------------------------------
    population_to_super = {}

    for pop, super_pop in zip(ref_pop, ref_super_pop):
        if pop not in population_to_super:
            population_to_super[pop] = super_pop

    continent_scores = {}

    for pop, score in population_scores.items():
        super_pop = population_to_super[pop]
        continent_scores[super_pop] = (
            continent_scores.get(super_pop, 0.0) + score
        )

    # ----------------------------------------------------
    # Find closest population
    # ----------------------------------------------------
    closest_population = max(
        population_scores,
        key=population_scores.get
    )

    confidence = population_scores[closest_population]

    # Distance to nearest reference individual
    dists = np.linalg.norm(
        ref_pcs - user_pcs[np.newaxis, :],
        axis=1
    )

    return {
        "pca_coordinates": user_pcs[:2].tolist(),

        "ancestry_proportions": {
            k: round(v, 4)
            for k, v in sorted(
                continent_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        },

        "closest_population": closest_population,

        "closest_populations": {
            k: round(v, 4)
            for k, v in sorted(
                population_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
        },

        "confidence": round(float(confidence), 3),

        "mean_neighbor_distance": round(float(dists.min()), 3),
    }

def handle_request(body: bytes) -> dict:
    model = load_model()
    calls = parse_upload(body, model)

    if len(calls) < MIN_SNPS_REQUIRED:
        return {
            "error": (
                f"Only {len(calls)} overlapping SNPs found in the uploaded file "
                f"(need at least {MIN_SNPS_REQUIRED}). "
                "For large raw exports, use a current browser so the file can be "
                "filtered before upload. Also check genome build (reference is hg19)."
            )
        }

    dosage, n_matched, n_dropped = build_feature_vector(calls, model)

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
        except Exception as exc:  # noqa: BLE001
            self._respond(500, {"error": f"Internal error: {exc}"})

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
