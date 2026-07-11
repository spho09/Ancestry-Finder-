"""
api/predict.py

Vercel Python serverless function for the Genetic Ancestry Estimator.

METHODOLOGY (see offline/build_reference_model.py for the training side)
--------------------------------------------------------------------------
Training data:  1000 Genomes Phase 3, 2,504 individuals, 26 populations
                 across 5 continental groups (AFR/AMR/EAS/EUR/SAS).
Test data:      the uploaded raw SNP file (a single, previously-unseen
                 genome, never used to build the reference model).

Inference is PCA-based similarity to reference populations, in three steps:
  1. Project the uploaded sample's genotypes into the same PCA space the
     reference panel was fit in (offline, once). This uses ancestry-
     informative SNPs -- markers whose allele frequency differs across
     populations because of demographic history: geographic isolation,
     migration bottlenecks, and genetic drift over the roughly 50,000+
     years since modern humans spread out of Africa. No single SNP is
     deterministic of ancestry; the *pattern* across thousands of them is.
  2. Compare the projected point against each of the 26 reference
     populations' precomputed centroids, using a Gaussian/RBF-style
     similarity kernel whose bandwidth is that population's own spread
     (see offline script for why). This produces a smooth probability
     over all 26 populations -- not a single winner -- so a genuinely
     admixed or intermediate sample naturally comes out as a mixture
     instead of being forced into one bucket.
  3. Aggregate the 26 population probabilities up to 5 continental
     ("ancestry_proportions") groups using the offline-computed
     population -> superpopulation mapping.

This is deliberately ONE primary inference model (RBF/Gaussian similarity
to precomputed centroids), not several competing heuristics stitched
together. It replaces an earlier k-nearest-neighbors vote over individual
reference samples, which tended toward overconfident, winner-take-all
results (e.g. "100% American") because a fixed-k vote doesn't account for
how tight or diffuse each population's own genetic variation is.

IMPORTANT SCIENTIFIC FRAMING
-----------------------------
This tool estimates genetic similarity to public reference populations.
It is NOT a determination of ethnicity (a social/cultural category) or of
exact genealogical ancestry. "Closest reference population" means
"most similar, among the 26 available," not "your ancestors came from
here" -- 1000 Genomes samples specific present-day populations, not a
complete map of human genetic variation.

Only depends on numpy at request time -- keeps the deployed bundle small
and cold starts fast. All population-level statistics (centroids, spread,
population->superpopulation mapping) are precomputed offline; this file
only loads, projects, compares, and returns.
"""

import json
import math
import os
from http.server import BaseHTTPRequestHandler

import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "reference_model.npz")
MIN_SNPS_REQUIRED = 500  # below this, results are too noisy to report

_model = None  # lazy-loaded, cached across warm invocations


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model() -> dict:
    """Load the precomputed reference model. Everything population-level
    (centroids, spread, superpopulation mapping) was computed offline in
    build_reference_model.py -- this function just reads it off disk."""
    global _model
    if _model is None:
        with np.load(MODEL_PATH, allow_pickle=False) as data:
            _model = {k: data[k] for k in data.files}
        snp_ids = _model["snp_ids"]
        _model["id_to_idx"] = {str(rsid): i for i, rsid in enumerate(snp_ids)}
    return _model


# ---------------------------------------------------------------------------
# Parsing the uploaded file
# ---------------------------------------------------------------------------

def _parse_23andme_line(line: str, id_to_idx: dict, calls: dict) -> None:
    parts = line.split("\t")
    if len(parts) < 4:
        parts = line.split(",")
    if len(parts) < 4:
        return
    rsid = parts[0]
    if id_to_idx.get(rsid) is None:
        return
    genotype = parts[3].replace("-", "")
    if len(genotype) == 2:
        calls[rsid] = (genotype[0], genotype[1])


def _parse_vcf_line(line: str, id_to_idx: dict, calls: dict) -> None:
    fields = line.split("\t")
    if len(fields) < 10:
        return
    rsid = fields[2]
    if id_to_idx.get(rsid) is None:
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
    """Parse only SNPs present in the reference panel (single pass, low
    memory). Supports 23andMe/AncestryDNA raw text and single-sample VCF,
    auto-detected from the first non-empty line."""
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
        if len(calls) >= len(id_to_idx):
            break  # every reference SNP has been found, no need to keep reading
    return calls


def build_feature_vector(calls: dict, model: dict):
    """Turn parsed genotype calls into a dosage vector aligned with the
    reference SNP order. SNPs the upload didn't cover are mean-imputed
    using the reference allele frequency (standard practice -- avoids
    biasing the projection toward zero for missing markers)."""
    ref_allele = model["ref_allele"]
    alt_allele = model["alt_allele"]
    allele_freq = model["allele_freq"]
    id_to_idx = model["id_to_idx"]

    dosage = 2 * allele_freq.copy()
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


# ---------------------------------------------------------------------------
# Inference: project -> compare -> aggregate -> confidence -> response
# ---------------------------------------------------------------------------

def project_sample(dosage: np.ndarray, model: dict) -> np.ndarray:
    """Project a genotype dosage vector into the reference PCA space.
    Single matrix multiply -- the PCA was fit offline; this is the only
    per-sample linear-algebra step left at request time."""
    pca_mean = model["pca_mean"]
    pca_components = model["pca_components"]
    return (dosage - pca_mean) @ pca_components.T


def compute_population_probabilities(user_pcs: np.ndarray, model: dict) -> dict:
    """Score the projected sample against every reference population using
    an RBF (Gaussian) similarity kernel centered at each population's
    precomputed centroid, with that population's own precomputed spread
    as the kernel bandwidth.

    This is the core replacement for the old k-NN vote. Conceptually it's
    a lightweight, isotropic Gaussian-mixture posterior: each population
    is treated as a spherical Gaussian cluster in PCA space (mean =
    centroid, std = spread), and we compute, under equal priors, how
    likely the sample is to belong to each one -- then normalize into a
    probability distribution over all 26 populations.

    Computed in log-space (log-sum-exp trick) purely for numerical
    stability -- mathematically this is exactly a softmax over
    negative squared, spread-normalized distances, one of the
    "statistically reasonable continuous similarity" approaches suggested
    for this refactor, combined with RBF-style per-population bandwidths.

    TODO(future upgrade): swap in a full ADMIXTURE-style model (explicit
    generative model of allele frequencies under k-way admixture) for a
    more rigorous proportion estimate than this PCA-centroid proxy, if
    runtime budget allows running it offline per-request precursor stats.
    """
    pop_codes = model["pop_codes"]
    pop_centroids = model["pop_centroids"]
    pop_spread = model["pop_spread"]

    dists = np.linalg.norm(pop_centroids - user_pcs[np.newaxis, :], axis=1)
    log_sim = -(dists ** 2) / (2 * pop_spread ** 2)

    # log-sum-exp normalization (numerically stable softmax)
    log_sim -= log_sim.max()
    weights = np.exp(log_sim)
    weights /= weights.sum()

    probabilities = {str(code): float(w) for code, w in zip(pop_codes, weights)}
    distances = {str(code): float(d) for code, d in zip(pop_codes, dists)}
    return probabilities, distances


def aggregate_superpopulations(pop_probabilities: dict, model: dict) -> dict:
    """Sum detailed-population probabilities up to the 5 continental
    groups, using the offline-precomputed population -> superpopulation
    mapping (no lookup logic recomputed here)."""
    pop_codes = model["pop_codes"]
    pop_to_super = model["pop_to_super"]
    pop_to_super_map = {str(p): str(s) for p, s in zip(pop_codes, pop_to_super)}

    continental = {}
    for pop, prob in pop_probabilities.items():
        sup = pop_to_super_map[pop]
        continental[sup] = continental.get(sup, 0.0) + prob
    return continental


def calculate_confidence(pop_probabilities: dict) -> dict:
    """Confidence based on the Shannon entropy of the population
    probability distribution, normalized to [0, 1] and inverted so higher
    = more confident. A sample that clusters tightly with one population
    has low entropy (high confidence); an admixed sample whose probability
    mass is spread across several populations has high entropy (low
    confidence) -- this falls out naturally from the distribution shape,
    rather than being a separate heuristic bolted on afterward.

    score = 1 - H(p) / H_max,  H_max = log(n_populations)  [max-entropy /
    fully-uniform case]

    Also reports the ratio between the top two populations' probabilities
    as a secondary, more intuitive signal ("how much stronger is the best
    match than the runner-up").
    """
    probs = np.array(list(pop_probabilities.values()))
    probs = probs[probs > 0]  # avoid log(0)
    n = len(pop_probabilities)

    entropy = -np.sum(probs * np.log(probs))
    max_entropy = math.log(n)
    score = 1.0 - (entropy / max_entropy if max_entropy > 0 else 0.0)
    score = float(np.clip(score, 0.0, 1.0))

    sorted_probs = sorted(pop_probabilities.values(), reverse=True)
    top1 = sorted_probs[0]
    top2 = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    top_ratio = float(top1 / top2) if top2 > 1e-9 else float("inf")

    if score >= 0.66:
        label = "High"
    elif score >= 0.38:
        label = "Medium"
    else:
        label = "Low"

    return {"label": label, "score": round(score, 3), "top_ratio": round(min(top_ratio, 99.0), 2)}


def build_response(
    user_pcs: np.ndarray,
    pop_probabilities: dict,
    pop_distances: dict,
    continental: dict,
    confidence: dict,
    n_matched: int,
    n_dropped: int,
    n_total: int,
) -> dict:
    """Assemble the final JSON-serializable response. Kept as a pure
    formatting step -- all the actual inference happens upstream, so this
    function does no math of its own."""
    sorted_pops = sorted(pop_probabilities.items(), key=lambda kv: kv[1], reverse=True)
    closest_population = sorted_pops[0][0]

    # "mean_neighbor_distance": distance to the 3 nearest population
    # centroids, averaged -- kept under the original field name for API
    # compatibility, redefined here as "how close is the sample to its
    # nearest neighboring reference populations" now that inference is
    # centroid-based rather than individual-sample k-NN.
    nearest_3 = sorted(pop_distances.values())[:3]
    mean_neighbor_distance = float(np.mean(nearest_3))

    sorted_continental = dict(
        sorted(continental.items(), key=lambda kv: kv[1], reverse=True)
    )

    return {
        "ancestry_proportions": {k: round(v, 4) for k, v in sorted_continental.items()},
        "closest_population": closest_population,
        "closest_populations": {k: round(v, 4) for k, v in sorted_pops},
        "confidence": confidence,
        "mean_neighbor_distance": round(mean_neighbor_distance, 3),
        "pca_coordinates": user_pcs[:2].tolist(),
        "population_distances": {k: round(v, 3) for k, v in pop_distances.items()},
        "n_snps_used": int(n_matched),
        "n_snps_total_in_reference": int(n_total),
        "n_allele_mismatches_dropped": int(n_dropped),
        "note": (
            "These results describe genetic similarity to public 1000 Genomes "
            "reference populations, based on ancestry-informative SNPs whose "
            "frequencies differ across populations due to demographic history "
            "(migration, geographic isolation, and genetic drift). This is not "
            "a determination of ethnicity, which is a social and cultural "
            "category, nor exact genealogical ancestry -- it reflects "
            "similarity to the specific populations sampled by 1000 Genomes, "
            "not a complete map of human genetic variation."
        ),
    }


# ---------------------------------------------------------------------------
# Request handling
# ---------------------------------------------------------------------------

def handle_request(body: bytes) -> dict:
    model = load_model()

    calls = parse_upload(body, model)
    if len(calls) < MIN_SNPS_REQUIRED:
        return {
            "error": (
                f"Only {len(calls)} overlapping SNPs found in the uploaded file "
                f"(need at least {MIN_SNPS_REQUIRED}). This usually means either "
                "the file isn't a supported format (23andMe/AncestryDNA raw text "
                "or single-sample VCF), or it uses a different genome build than "
                "the reference panel (this model expects hg19)."
            ),
            "error_type": "insufficient_overlap",
        }

    dosage, n_matched, n_dropped = build_feature_vector(calls, model)
    if n_matched < MIN_SNPS_REQUIRED:
        return {
            "error": (
                f"Only {n_matched} of your SNPs matched the reference panel's "
                f"expected alleles (need at least {MIN_SNPS_REQUIRED}). This can "
                "happen with a genome-build mismatch (this model expects hg19) "
                "or a corrupted/malformed export."
            ),
            "error_type": "genome_build_mismatch",
        }

    user_pcs = project_sample(dosage, model)
    pop_probabilities, pop_distances = compute_population_probabilities(user_pcs, model)
    continental = aggregate_superpopulations(pop_probabilities, model)
    confidence = calculate_confidence(pop_probabilities)

    return build_response(
        user_pcs=user_pcs,
        pop_probabilities=pop_probabilities,
        pop_distances=pop_distances,
        continental=continental,
        confidence=confidence,
        n_matched=n_matched,
        n_dropped=n_dropped,
        n_total=len(model["snp_ids"]),
    )


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length == 0:
                self._respond(400, {"error": "Empty upload.", "error_type": "malformed_upload"})
                return
            body = self.rfile.read(length)
            result = handle_request(body)
            status = 400 if "error" in result else 200
            self._respond(status, result)
        except Exception as exc:  # noqa: BLE001
            self._respond(500, {"error": f"Internal error: {exc}", "error_type": "server_error"})

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
