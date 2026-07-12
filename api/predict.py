"""
api/predict.py

Vercel Python serverless function for the Genetic Ancestry Estimator.

METHODOLOGY (see offline/build_reference_model.py for the training side)
--------------------------------------------------------------------------
Training data:  1000 Genomes Phase 3, 2,504 individuals, 26 populations
                 across 5 continental groups (AFR/AMR/EAS/EUR/SAS).
Test data:      the uploaded raw SNP file (a single, previously-unseen
                 genome, never used to build the reference model).

HYBRID INFERENCE PIPELINE
--------------------------
This is a two-stage pipeline: PCA narrows the search space, then an
allele-frequency-based genotype likelihood refines the result within that
narrowed space. Neither stage alone is the "whole" model -- PCA is a fast,
approximate first pass; the likelihood stage is what actually reasons
about genotypes directly.

  1. PROJECT: project the uploaded sample's genotypes into the reference
     PCA space (fit offline, once), using ancestry-informative SNPs --
     markers whose allele frequency differs across populations because of
     demographic history: geographic isolation, migration, and genetic
     drift over the ~50,000+ years since modern humans spread out of
     Africa.
  2. FIND CANDIDATES: score all 26 reference populations by Gaussian/RBF
     similarity to the sample's PCA position (bandwidth = that
     population's own spread), take the top-K as candidates. This is a
     fast, coarse filter -- it doesn't need to be exact, just needs to
     not discard the right answer.
  3. REFINE WITH ALLELE FREQUENCIES: for each candidate only, compute how
     well the sample's *observed* genotypes fit that population's actual
     allele frequencies, under Hardy-Weinberg equilibrium
     (P(0 copies)=(1-r)^2, P(1)=2r(1-r), P(2)=r^2, r = population allele
     frequency at that SNP). This is the same generative idea used by
     ADMIXTURE-style tools (and, conceptually, by the AEON reference
     project this was benchmarked against) -- reasoning directly about
     genotype probabilities, not just distance in a linear projection.
     Unlike a full ADMIXTURE/AEON run, this does NOT fit a continuous
     admixture-proportion vector via gradient-based optimization; it
     scores each of the K candidate populations once, in closed form, via
     vectorized NumPy -- no optimizer, no autodiff, no extra dependency.
  4. COMBINE: combine the PCA prior and the allele-frequency likelihood in
     log-space (equivalent to Bayes' rule: log posterior = log prior +
     log likelihood), then softmax-normalize over the candidate set into
     final probabilities.
  5. AGGREGATE: sum detailed-population probabilities up to 5 continental
     groups using the offline-precomputed population -> superpopulation
     mapping.

WHY THIS HELPS THE "EVERYTHING LOOKS AMERICAN" PROBLEM
---------------------------------------------------------
1000 Genomes' AMR populations (MXL/PUR/CLM/PEL) are themselves admixed and
have the widest PCA spread of any group in the panel (a wide,
diffuse cluster in PCA space). PCA distance alone can't tell "genuinely
similar to admixed American populations" apart from "not a strong match
for anything else available, and AMR's kernel is wide enough to accept
it anyway" -- which matters a lot for genomes from regions 1000 Genomes
doesn't sample (Central Asia, Siberia, Middle East, North Africa): they
can get pulled toward AMR by default. The allele-frequency likelihood is
a different signal -- it measures fit to actual genotype patterns, not
proximity in a linear projection -- so it doesn't inherit that same bias,
and pulls the combined estimate away from AMR when the genotype evidence
doesn't actually support it.

IMPORTANT SCIENTIFIC FRAMING
-----------------------------
This tool estimates genetic similarity to public reference populations.
It is NOT a determination of ethnicity (a social/cultural category) or of
exact genealogical ancestry. A genome whose true origin isn't well
represented among the 26 available populations will still be assigned to
whichever are the closest available matches -- that's a limitation of
reference panel coverage, not a claim about the person's actual heritage.

Only depends on numpy at request time -- keeps the deployed bundle small
and cold starts fast. All population-level statistics (centroids, spread,
allele frequencies, population->superpopulation mapping) are precomputed
offline; this file only loads, projects, scores, combines, and returns.
"""

import json
import math
import os
from http.server import BaseHTTPRequestHandler

import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "reference_model.npz")
MIN_SNPS_REQUIRED = 500  # below this, results are too noisy to report

# How many of the 26 reference populations survive the PCA stage to be
# scored by the (more expensive, more informative) allele-frequency
# likelihood stage. Large enough that the true population is essentially
# never excluded by PCA's coarse first pass; small enough to keep the
# likelihood stage's cost bounded and its output interpretable (we don't
# want to report tiny nonzero "candidate" probability for populations PCA
# already confidently rules out).
TOP_K_CANDIDATES = 12

# How strongly the allele-frequency likelihood can move the final
# probabilities relative to the PCA prior. Chosen empirically: per-SNP
# mean log-likelihood differences between genuinely close populations are
# small (~0.01-0.05) while differences between clearly-wrong populations
# are much larger (~0.2-0.4) -- see module tests. A weight of 100 makes
# the likelihood stage decisive between close candidates without letting
# it swing wildly on noise from a handful of SNPs.
LIKELIHOOD_WEIGHT = 2.0

_model = None  # lazy-loaded, cached across warm invocations


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model() -> dict:
    """Load the precomputed reference model. Everything population-level
    (centroids, spread, allele frequencies, superpopulation mapping) was
    computed offline in build_reference_model.py -- this function just
    reads it off disk."""
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

# Only single-base SNP alleles are usable by this model (no indels).
# Filtering to this set also doubles as no-call handling: AncestryDNA
# marks no-calls as "0" per allele, 23andMe marks them as "--" (which
# becomes two "-" characters once split) -- neither "0" nor "-" is in
# this set, so both formats' no-calls are naturally rejected by the same
# check, with no special-casing needed.
_VALID_BASES = {"A", "C", "G", "T"}


def _parse_raw_text_line(line: str, id_to_idx: dict, calls: dict) -> None:
    """Parses one data line from a 23andMe-style OR AncestryDNA-style raw
    text export.

    23andMe format -- 4 columns, alleles COMBINED in one field:
        rsid    chromosome  position    genotype
        rs4477212   1       82154       AA

    AncestryDNA format -- 5 columns, alleles in TWO SEPARATE fields:
        rsid    chromosome  position    allele1 allele2
        rs4477212   1       82154       A       A

    BUG THIS FIXES: the previous version only handled the 4-column case.
    For a 5-column AncestryDNA line, parts[3] is a single character (just
    allele1), so the old `len(genotype) == 2` check always failed and the
    line was silently dropped -- for every row in a real AncestryDNA
    file, producing ~0 matched SNPs.
    """
    parts = line.split("\t")
    if len(parts) < 4:
        parts = line.split(",")
    if len(parts) < 4:
        return

    rsid = parts[0].strip()
    if id_to_idx.get(rsid) is None:
        return

    if len(parts) >= 5:
        # AncestryDNA-style: two separate single-allele columns
        a1 = parts[3].strip().upper()
        a2 = parts[4].strip().upper()
    else:
        # 23andMe-style: alleles combined into one column
        combined = parts[3].strip().upper()
        if len(combined) != 2:
            return
        a1, a2 = combined[0], combined[1]

    if a1 in _VALID_BASES and a2 in _VALID_BASES:
        calls[rsid] = (a1, a2)


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


def parse_upload(body: bytes, model: dict):
    """Parse only SNPs present in the reference panel (single pass, low
    memory). Supports 23andMe/AncestryDNA raw text and single-sample VCF,
    auto-detected from the first non-empty line.

    Returns (calls, diagnostics). `calls` only ever contains rsIDs that
    matched the reference panel (matching happens inline, during parsing,
    for memory efficiency on large files) -- so a separate `n_parsed`
    counter is tracked here too, to distinguish "the file had rows but
    none matched the reference panel" from "the file had almost no
    syntactically valid rows at all." Those point to different root
    causes (rsID/build mismatch vs. a parser/format bug) and mixing them
    up made this exact bug harder to diagnose."""
    id_to_idx = model["id_to_idx"]
    calls = {}
    is_vcf = False
    header_checked = False
    n_parsed = 0
    parsed_sample = []

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

        # count syntactically-valid data rows regardless of reference
        # match, for the "parser bug vs. no overlap" diagnostic below
        parts = line.split("\t") if "\t" in line else line.split(",")
        if len(parts) >= 4 and parts[0].strip():
            n_parsed += 1
            if len(parsed_sample) < 20:
                parsed_sample.append(parts[0].strip())

        n_before = len(calls)
        if is_vcf:
            _parse_vcf_line(line, id_to_idx, calls)
        else:
            _parse_raw_text_line(line, id_to_idx, calls)

        if len(calls) >= len(id_to_idx):
            break  # every reference SNP has been found, no need to keep reading

    diagnostics = {
        "n_parsed": n_parsed,
        "n_reference": len(id_to_idx),
        "n_matching": len(calls),
        "match_percentage": round((len(calls) / n_parsed * 100) if n_parsed else 0.0, 2),
        "sample_parsed_rsids": parsed_sample,
        "sample_matching_rsids": list(calls.keys())[:20],
    }
    return calls, diagnostics


def _log_overlap_diagnostics(diagnostics: dict) -> None:
    """Debug output requested when overlap is suspiciously low. Printed
    to stdout, which lands in Vercel's function logs (Project -> Logs in
    the Vercel dashboard, or `vercel logs` from the CLI)."""
    print(f"Parsed SNPs: {diagnostics['n_parsed']}")
    print(f"Reference SNPs: {diagnostics['n_reference']}")
    print(f"Matching SNPs: {diagnostics['n_matching']}")
    print(f"Match percentage: {diagnostics['match_percentage']}%")
    print(f"First 20 parsed rsIDs: {diagnostics['sample_parsed_rsids']}")
    print(f"First 20 matching rsIDs: {diagnostics['sample_matching_rsids']}")


def build_feature_vector(calls: dict, model: dict):
    """Turn parsed genotype calls into a dosage vector aligned with the
    reference SNP order, plus a boolean mask of which SNPs were actually
    observed (as opposed to mean-imputed). The mask matters for the
    allele-frequency likelihood stage: imputed entries carry no real
    genotype information about this specific sample and must be excluded
    from the likelihood calculation, or the likelihood would just be
    scoring the reference panel's own average against itself."""
    ref_allele = model["ref_allele"]
    alt_allele = model["alt_allele"]
    allele_freq = model["allele_freq"]
    id_to_idx = model["id_to_idx"]

    n_snps = len(allele_freq)
    dosage = 2 * allele_freq.copy()
    observed_mask = np.zeros(n_snps, dtype=bool)
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
        observed_mask[idx] = True
        n_matched += 1

    return dosage, observed_mask, n_matched, n_ambiguous_dropped


# ---------------------------------------------------------------------------
# Stage 1: PCA projection
# ---------------------------------------------------------------------------

def project_sample(dosage: np.ndarray, model: dict) -> np.ndarray:
    """Project a genotype dosage vector into the reference PCA space.
    Single matrix multiply -- the PCA was fit offline; this is the only
    per-sample linear-algebra step needed for the PCA stage."""
    pca_mean = model["pca_mean"]
    pca_components = model["pca_components"]
    return (dosage - pca_mean) @ pca_components.T


# ---------------------------------------------------------------------------
# Stage 2: find candidate populations via PCA (coarse, fast filter)
# ---------------------------------------------------------------------------

def find_candidate_populations(user_pcs: np.ndarray, model: dict, top_k: int = TOP_K_CANDIDATES):
    """Score every reference population by Gaussian/RBF similarity to the
    sample's PCA position (bandwidth = that population's own spread --
    see offline script for why a per-population bandwidth matters), then
    keep only the top_k. Returns:
      - pca_log_prior: dict {pop_code: log(softmax similarity)} for ALL
        26 populations (used for the confidence/debug view and for the
        candidates' prior term in stage 4)
      - pca_distances: dict {pop_code: euclidean distance} for ALL 26
        (kept for the frontend's PCA visualization, unchanged from before)
      - candidate_codes: the top_k population codes that proceed to the
        allele-frequency refinement stage
    """
    pop_codes = model["pop_codes"]
    pop_centroids = model["pop_centroids"]
    pop_spread = model["pop_spread"]

    dists = np.linalg.norm(pop_centroids - user_pcs[np.newaxis, :], axis=1)
    log_sim = -(dists ** 2) / (2 * pop_spread ** 2)

    # log-sum-exp normalization (numerically stable softmax)
    log_sim_shifted = log_sim - log_sim.max()
    weights = np.exp(log_sim_shifted)
    weights /= weights.sum()

    pca_prior_prob = {str(code): float(w) for code, w in zip(pop_codes, weights)}
    pca_log_prior = {str(code): float(np.log(max(w, 1e-300))) for code, w in pca_prior_prob.items()}
    pca_distances = {str(code): float(d) for code, d in zip(pop_codes, dists)}

    ranked = sorted(pca_prior_prob.items(), key=lambda kv: kv[1], reverse=True)
    candidate_codes = [code for code, _ in ranked[:top_k]]

    return pca_prior_prob, pca_log_prior, pca_distances, candidate_codes


def compute_combination_prior(pca_distances: dict, candidate_codes: list, model: dict) -> dict:
    """A second, GENTLER PCA-based prior, used only in the final
    combination step (combine_pca_and_likelihood_scores) -- distinct from
    the per-population-spread-normalized prior above, which is fine for
    candidate SELECTION but actively wrong to use as a final weighting
    term. Here's why these need to be different:

    BUG THIS FIXES: the harsh, per-population-spread RBF divides by that
    population's own spread^2. Tight, historically isolated populations
    (e.g. CHB/CHS/KHV, spread ~10-13) have a very narrow "acceptance
    window" -- a sample needs to sit almost exactly at their centroid to
    score well. Diffuse, admixed populations (e.g. MXL/ASW, spread
    ~36-49) have a much wider one. For an in-training-set sample (which
    sits at ~zero distance from its own true population's centroid) this
    doesn't matter. For a genuinely new, real-world upload -- which
    normally sits at a real, nonzero distance from every centroid -- this
    bandwidth disparity can completely dominate: a tight population that
    is geometrically the CLOSEST match can still get an almost-zero prior
    simply for not being close enough, while a diffuse population that is
    much farther away gets a comfortable one. This is disproportionate to
    the actual signal and can overwhelm the (usually more reliable)
    allele-frequency likelihood stage entirely -- confirmed on a real
    AncestryDNA upload where the true ancestry was ~100% East Asian:
    KHV/CHS/CHB were the 3 geometrically closest populations of all 26,
    yet the harsh prior ranked them dead last among the 10 PCA
    candidates, while the allele-frequency likelihood (unaffected by this
    issue) correctly still favored them.

    The fix: use ONE SHARED bandwidth (the median spread across all 26
    populations, precomputed offline as `global_bandwidth`) for this
    prior, instead of each population's own idiosyncratic spread. This
    keeps real distance information in the prior (a genuinely closer
    population still gets a higher prior than a farther one) without the
    tight-vs-diffuse disparity distorting it by orders of magnitude. The
    per-population-spread version is still used, unchanged, for candidate
    SELECTION (find_candidate_populations) and for the frontend's
    "population_distances" debug view -- both of those are fine with it,
    since selection only needs good recall (not excluding the true
    answer), which it already has.
    """
    global_bandwidth = float(model["global_bandwidth"])
    log_sim = {
        code: -(pca_distances[code] ** 2) / (2 * global_bandwidth ** 2)
        for code in candidate_codes
    }
    m = max(log_sim.values())
    log_norm = m + math.log(sum(math.exp(v - m) for v in log_sim.values()))
    return {code: v - log_norm for code, v in log_sim.items()}


# ---------------------------------------------------------------------------
# Stage 3: refine candidates with an allele-frequency genotype likelihood
# ---------------------------------------------------------------------------

def compute_population_likelihoods(dosage: np.ndarray, observed_mask: np.ndarray, candidate_codes: list, model: dict) -> dict:
    """For each candidate population, compute the mean per-SNP
    log-likelihood of the sample's OBSERVED genotypes under that
    population's allele frequencies, assuming Hardy-Weinberg equilibrium:
        P(0 copies of alt allele) = (1-r)^2
        P(1 copy)                 = 2 r (1-r)
        P(2 copies)               = r^2
    where r is that population's allele frequency at that SNP.

    This is the piece conceptually borrowed from AEON: reasoning directly
    about genotype probabilities from allele frequencies, rather than
    relying only on PCA distance. Unlike AEON's continuous admixture-
    proportion fit (gradient-based optimization over a Dirichlet-
    distributed mixture), this scores each discrete candidate population
    once, via a single vectorized NumPy expression per candidate -- no
    optimizer, no iteration, safely within a serverless time budget.

    Mean (not summed) log-likelihood is used so the score doesn't depend
    on how many SNPs happened to be observed in a given upload -- a
    sparse consumer SNP-chip export and a dense imputed VCF should be
    comparable on this scale.
    """
    pop_codes = list(model["pop_codes"])
    pop_allele_freq = model["pop_allele_freq"]

    obs_idx = np.where(observed_mask)[0]
    d_obs = dosage[obs_idx].astype(np.int64)

    mean_loglik = {}
    for code in candidate_codes:
        p_idx = pop_codes.index(code)
        r = pop_allele_freq[p_idx, obs_idx]
        pr0 = (1 - r) ** 2
        pr1 = 2 * r * (1 - r)
        pr2 = r ** 2
        pr_stack = np.stack([pr0, pr1, pr2], axis=1)
        selected = pr_stack[np.arange(len(d_obs)), d_obs]
        loglik = np.sum(np.log(np.clip(selected, 1e-12, None)))
        mean_loglik[code] = float(loglik / max(len(d_obs), 1))

    return mean_loglik


# ---------------------------------------------------------------------------
# Stage 4: combine PCA prior and allele-frequency likelihood
# ---------------------------------------------------------------------------

def combine_pca_and_likelihood_scores(combination_prior: dict, mean_loglik: dict, candidate_codes: list) -> dict:
    """Bayesian-style combination in log space: log(posterior) is
    proportional to log(prior) + log(likelihood). `combination_prior`
    should be the GENTLE, shared-bandwidth prior from
    compute_combination_prior -- not the harsh per-population-spread
    prior from find_candidate_populations, which is only appropriate for
    candidate selection (see that function's docstring). Candidates
    outside the PCA-selected set implicitly get zero posterior -- they
    were pruned in stage 2. Softmax-normalizing the combined scores over
    just the candidate set turns them into a proper probability
    distribution.
    """
    combined_log_score = {
        code: combination_prior[code] + LIKELIHOOD_WEIGHT * mean_loglik[code]
        for code in candidate_codes
    }
    scores = np.array([combined_log_score[c] for c in candidate_codes])
    scores -= scores.max()  # numerically stable softmax
    weights = np.exp(scores)
    weights /= weights.sum()

    return {code: float(w) for code, w in zip(candidate_codes, weights)}


# ---------------------------------------------------------------------------
# Aggregation and confidence
# ---------------------------------------------------------------------------

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
    """Confidence based on the Shannon entropy of the (post-refinement)
    population probability distribution, normalized to [0, 1] and
    inverted so higher = more confident. A sample that clusters tightly
    with one population has low entropy (high confidence); a sample whose
    probability mass is spread across several populations has high
    entropy (low confidence) -- this falls out of the distribution shape
    itself, rather than being a separate heuristic.

    score = 1 - H(p) / H_max,  H_max = log(n_candidates)  [uniform-over-
    candidates case, i.e. "no idea which of the plausible candidates"]

    Also reports the ratio between the top two populations' probabilities
    as a secondary, more intuitive signal.
    """
    probs = np.array(list(pop_probabilities.values()))
    probs = probs[probs > 0]
    n = len(pop_probabilities)

    entropy = -np.sum(probs * np.log(probs))
    max_entropy = math.log(n) if n > 1 else 1.0
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


# ---------------------------------------------------------------------------
# Response assembly
# ---------------------------------------------------------------------------

def build_response(
    user_pcs: np.ndarray,
    pop_probabilities: dict,
    pca_distances: dict,
    continental: dict,
    confidence: dict,
    candidate_codes: list,
    n_matched: int,
    n_dropped: int,
    n_total: int,
) -> dict:
    """Assemble the final JSON-serializable response. Kept as a pure
    formatting step -- all the actual inference happens upstream."""
    sorted_pops = sorted(pop_probabilities.items(), key=lambda kv: kv[1], reverse=True)
    closest_population = sorted_pops[0][0]

    # distance to the 3 nearest PCA candidate centroids, averaged -- kept
    # under the original field name for API compatibility
    nearest_3 = sorted(pca_distances[c] for c in candidate_codes)[:3]
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
        "population_distances": {k: round(pca_distances[k], 3) for k in candidate_codes},
        "n_snps_used": int(n_matched),
        "n_snps_total_in_reference": int(n_total),
        "n_allele_mismatches_dropped": int(n_dropped),
        "note": (
            "These results describe genetic similarity to public 1000 Genomes "
            "reference populations, estimated in two stages: a PCA projection "
            "narrows the search to the most plausible reference populations, "
            "then an allele-frequency-based genotype likelihood (how well the "
            "sample's observed genotypes fit each candidate population's actual "
            "allele frequencies) refines the final probabilities. This is not a "
            "determination of ethnicity, which is a social and cultural "
            "category, nor exact genealogical ancestry. If your genome's true "
            "origin isn't well represented among the 26 populations in this "
            "reference panel (1000 Genomes has no Central Asian, Siberian, "
            "Middle Eastern, or North African populations, for example), the "
            "closest available matches will still be returned -- read this as "
            "'most similar to the following available reference populations,' "
            "not as a complete map of your ancestry."
        ),
    }


# ---------------------------------------------------------------------------
# Request handling
# ---------------------------------------------------------------------------

def handle_request(body: bytes) -> dict:
    model = load_model()

    calls, diagnostics = parse_upload(body, model)
    if len(calls) < MIN_SNPS_REQUIRED:
        _log_overlap_diagnostics(diagnostics)
        return {
            "error": (
                f"Only {len(calls)} overlapping SNPs found in the uploaded file "
                f"(need at least {MIN_SNPS_REQUIRED}). This usually means either "
                "the file isn't a supported format (23andMe/AncestryDNA raw text "
                "or single-sample VCF), or it uses a different genome build than "
                "the reference panel (this model expects hg19)."
            ),
            "error_type": "insufficient_overlap",
            "diagnostics": diagnostics,
        }

    dosage, observed_mask, n_matched, n_dropped = build_feature_vector(calls, model)
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

    # Stage 1: PCA projection
    user_pcs = project_sample(dosage, model)

    # Stage 2: PCA-based candidate narrowing (coarse, fast filter -- the
    # harsh per-population-spread prior is fine here, it only needs good
    # recall, which it has)
    pca_prior_prob, pca_log_prior, pca_distances, candidate_codes = find_candidate_populations(user_pcs, model)

    # Stage 3: allele-frequency genotype likelihood, candidates only
    mean_loglik = compute_population_likelihoods(dosage, observed_mask, candidate_codes, model)

    # Stage 4: combine into final population probabilities. Uses a
    # GENTLER, shared-bandwidth prior here (not pca_log_prior from stage
    # 2) -- see compute_combination_prior's docstring for why the harsh
    # per-population-spread prior is wrong for this specific step.
    combination_prior = compute_combination_prior(pca_distances, candidate_codes, model)
    pop_probabilities = combine_pca_and_likelihood_scores(combination_prior, mean_loglik, candidate_codes)

    # Stage 5: aggregate + confidence
    continental = aggregate_superpopulations(pop_probabilities, model)
    confidence = calculate_confidence(pop_probabilities)

    return build_response(
        user_pcs=user_pcs,
        pop_probabilities=pop_probabilities,
        pca_distances=pca_distances,
        continental=continental,
        confidence=confidence,
        candidate_codes=candidate_codes,
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
