# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Code fingerprint recorded with each run, answering "did every run use the same code?".

Each trial records SHA-256 digests of the files whose contents decide behaviour; a
launcher can record the same at submit time. Comparing the two, or comparing across
runs, makes code divergence visible in the artifacts however it happened (pull, editor,
a half-applied patch). Computed once per process and cached.

    from clm_harness.utils.provenance import code_fingerprint
    fp = code_fingerprint()          # {"harness": "ab12…", …, "combined": "9f3c…"}
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

#: Core files whose contents decide agent behaviour, reported individually so a
#: mismatch says where two runs differ. `env.py` holds the observation/edit machinery.
_WATCHED = {
    "harness": _ROOT / "clm_agent" / "harness.py",
    "context_env": _ROOT / "context_env" / "env.py",
}

_CACHE: dict[str, str] | None = None


def code_fingerprint() -> dict[str, str]:
    """SHA-256 (first 12 hex chars) per watched file, plus a combined digest.

    Truncated to 12 characters: this is for equality comparison between trials,
    not for anything adversarial, and a full 64-char hash times four keys
    would be a visible fraction of a small usage.json.

    A missing file records ``"absent"`` rather than raising. Provenance recording must
    never be the reason a trial fails -- the whole point is that it is a passive
    witness, and a witness that can crash the run is worse than none.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    out: dict[str, str] = {}
    combined = hashlib.sha256()
    for name, path in sorted(_WATCHED.items()):
        try:
            data = path.read_bytes()
        except Exception:
            out[name] = "absent"
            combined.update(b"\0absent\0")
            continue
        out[name] = hashlib.sha256(data).hexdigest()[:12]
        combined.update(data)
    out["combined"] = combined.hexdigest()[:12]
    # Package digest over every .py under clm_harness/, sorted. Behaviour also lives in
    # other modules (budget.py, edit_gate.py, ...), so this covers all of them. The
    # per-file keys say WHERE two runs differ; `package` says WHETHER.
    pkg = hashlib.sha256()
    for f in sorted(_ROOT.rglob("*.py")):
        try:
            pkg.update(f.relative_to(_ROOT).as_posix().encode())
            pkg.update(b"\0")
            pkg.update(f.read_bytes())
        except Exception:
            pkg.update(b"\0absent\0")
    out["package"] = pkg.hexdigest()[:12]
    _CACHE = out
    return out


def main() -> int:
    """`python -m clm_harness.utils.provenance` — what a launcher records at submit time."""
    import json

    print(json.dumps(code_fingerprint(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
