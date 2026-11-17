"""End to end: match data -> out/TheObservers.csv, then the organisers' format validation.

    python run.py                       # default method
    python run.py --method baseline50   # the organisers' published floor, recomputed by us

Never retrains anything. Deterministic: the same data bundle always gives the same bytes, and the
sha256 printed at the end is the one to quote in the technical document.
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from possession import baseline, config, write  # noqa: E402


def method_baseline50():
    """M0: the organisers' baseline_smooth50, recomputed through our loader (pipeline check)."""
    return baseline.baseline_labels(baseline.baseline_codes(), 50)


def method_v1(use_tags=True):
    """M1: rule-based evidence + kickoff detection + Smart Tagging prior + segmental Viterbi."""
    from possession import io_atd, io_tagging, v1
    match = io_atd.load_match()
    tags = io_tagging.load() if use_tags else None
    codes, dbg = v1.predict(match, tags, use_tags=use_tags, return_debug=True)
    describe(codes, match, dbg)
    return codes


def method_v1_notags():
    return method_v1(use_tags=False)


def method_v2(use_tags=True):
    """M2: emissions learned on degraded IDSSE + kickoffs + tag prior + CV-tuned decoder."""
    from possession import io_atd, io_tagging, v2
    match = io_atd.load_match()
    tags = io_tagging.load() if use_tags else None
    codes, dbg = v2.predict(match, tags, use_tags=use_tags, return_debug=True)
    print("evidence scale (classifier nats per v1 nat): {:.2f}".format(dbg["evidence_scale"]))
    describe(codes, match, dbg)
    return codes


def method_v2_notags():
    return method_v2(use_tags=False)


def method_v3(use_tags=True):
    """M3: larger classifier + v1/v2 evidence ensemble, settings chosen on IDSSE nested CV."""
    from possession import io_atd, io_tagging, v3
    match = io_atd.load_match()
    tags = io_tagging.load() if use_tags else None
    codes, dbg = v3.predict(match, tags, use_tags=use_tags, return_debug=True)
    print("evidence scale {:.2f}  rule weight {:.2f}".format(dbg["evidence_scale"],
                                                             dbg["rule_weight"]))
    describe(codes, match, dbg)
    return codes


def method_v3_notags():
    return method_v3(use_tags=False)


def method_v4(use_tags=True):
    """v4: v2 pipeline on feature version 2 (touch detection + slowdown), IDSSE nested CV."""
    from possession import io_atd, io_tagging, v4
    match = io_atd.load_match()
    tags = io_tagging.load() if use_tags else None
    codes, dbg = v4.predict(match, tags, use_tags=use_tags, return_debug=True)
    print("evidence scale {:.2f}".format(dbg["evidence_scale"]))
    describe(codes, match, dbg)
    return codes


def method_v4_notags():
    return method_v4(use_tags=False)


def method_v5(use_tags=True):
    """v5 candidate: v4 pipeline with neural sequence emissions (out/models_v5, scripts/train_nn.py)."""
    from possession import io_atd, io_tagging, v5
    match = io_atd.load_match()
    tags = io_tagging.load() if use_tags else None
    codes, dbg = v5.predict(match, tags, use_tags=use_tags, return_debug=True)
    print("evidence scale {:.2f}".format(dbg["evidence_scale"]))
    describe(codes, match, dbg)
    return codes


def method_v6(use_tags=True):
    """v6 candidate: v5 with members trained on the dead-time ball degrader (out/models_v6)."""
    from possession import io_atd, io_tagging, v6
    match = io_atd.load_match()
    tags = io_tagging.load() if use_tags else None
    codes, dbg = v6.predict(match, tags, use_tags=use_tags, return_debug=True)
    print("evidence scale {:.2f}".format(dbg["evidence_scale"]))
    describe(codes, match, dbg)
    return codes


def method_v51(use_tags=True, version="v51"):
    """v5.1: v5's members + the Law-9 out-of-play override, decoder re-tuned with it. Weights and
    config are committed in models/v51 (IDSSE-trained), so this never retrains."""
    from possession import io_atd, io_tagging, v5
    match = io_atd.load_match()
    tags = io_tagging.load() if use_tags else None
    model_dir = (os.environ.get("POSSESSION_{}_DIR".format(version.upper())) or
                 os.path.join(config.REPO_DIR, "models", version))
    codes, dbg = v5.predict(match, tags, use_tags=use_tags, return_debug=True, model_dir=model_dir)
    print("evidence scale {:.2f}".format(dbg["evidence_scale"]))
    describe(codes, match, dbg)
    return codes


def method_v52(use_tags=True):
    """v5.2: v5.1 with the decoder tuned on the mean spellF1 over the 2 / 1 / 0.5 s conventions
    (models/v52/v5.json; members are v5.1's committed weights)."""
    return method_v51(use_tags, version="v52")


def method_v53(use_tags=True):
    """v5.3 candidate: v5.2 + broadcast ball fill of tactical gaps (possession/ballfill.py)."""
    return method_v51(use_tags, version="v53")


def method_v54(use_tags=True):
    """v5.4: v5.2 + broadcast replay evidence on D (possession/replay.py, frozen log-ratio)."""
    return method_v51(use_tags, version="v54")


def method_v55(use_tags=True):
    """v5.5: v5.4 with two more members (switch-weighted GRU, 12.5 fps GRU) and their mean-tuned
    decoder (models/v55)."""
    return method_v51(use_tags, version="v55")


def describe(codes, match, dbg):
    """Ground-truth-free plausibility summary of a prediction (aggregates only)."""
    import numpy as np
    g = match.graded
    share = {k: float(np.mean(codes[g] == i)) for i, k in enumerate("HAD")}
    n_poss, n_dead, lengths = 0, 0, []
    for _, s, e in match.half_blocks():
        c = codes[s:e]
        b = np.flatnonzero(np.diff(c)) + 1
        starts, stops = np.r_[0, b], np.r_[b, len(c)]
        for a, z in zip(starts, stops):
            if c[a] == 2:
                n_dead += 1
            else:
                n_poss += 1
                lengths.append(z - a)
    print("kickoffs detected: {} (pre-kickoff spans: {})  speed threshold {:.2f} m/s  "
          "attack sign {}".format(len(dbg["kickoffs"]), len(dbg["pre_kickoff"]),
                                  dbg["speed_threshold"], dbg["attack_sign"]))
    print("graded shares  H {H:.3f}  A {A:.3f}  D {D:.3f}".format(**share))
    print("possessions {}  mean {:.1f}s  median {:.1f}s  dead-ball stretches {}".format(
        n_poss, np.mean(lengths) / 25, np.median(lengths) / 25, n_dead), flush=True)


METHODS = {
    "baseline50": method_baseline50,
    "v1": method_v1,
    "v1_notags": method_v1_notags,
    "v2": method_v2,
    "v2_notags": method_v2_notags,
    "v3": method_v3,
    "v3_notags": method_v3_notags,
    "v4": method_v4,
    "v4_notags": method_v4_notags,
    "v5": method_v5,
    "v6": method_v6,
    "v51": method_v51,
    "v52": method_v52,
    "v53": method_v53,
    "v54": method_v54,
    "v55": method_v55,
}
DEFAULT_METHOD = "v54"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", choices=sorted(METHODS), default=DEFAULT_METHOD)
    parser.add_argument("--out", default=os.path.join(config.OUT_DIR, config.SUBMISSION_NAME))
    args = parser.parse_args()

    labels = METHODS[args.method]()
    digest = write.write_submission(args.out, labels)
    print("method {}  ->  {}".format(args.method, args.out))
    print("sha256 {}\n".format(digest), flush=True)

    validator = os.path.join(config.MTAH_REPO_DIR, "evaluation", "validate_submission.py")
    return subprocess.run([sys.executable, validator, args.out]).returncode


if __name__ == "__main__":
    sys.exit(main())
