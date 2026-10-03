#!/usr/bin/env bash
# Initialize the third-party submodules this study needs, shallowly, by group.
#
#   scripts/setup_third_party.sh core        # PoS loop + LOCA-bench + ALFWorld + SKILL.state runtime + Supersede + LongMemEval
#   scripts/setup_third_party.sh methods     # every comparison method (Scroll, VISTA, RLM, SelfCompact, ACM, BB-WM, ...)
#   scripts/setup_third_party.sh benchmarks  # every benchmark except the optional heavy ones
#   scripts/setup_third_party.sh references  # read-only references (no license; do not copy code)
#   scripts/setup_third_party.sh optional    # STATE-Bench and BEAM (BEAM is ~4 GB)
#   scripts/setup_third_party.sh all         # everything above
#
# Pinned commits live in the superproject; see third_party/README.md for roles and licenses.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

core=(
  third_party/methods/pos
  third_party/methods/skill-state-runtime
  third_party/benchmarks/loca-bench
  third_party/benchmarks/alfworld
  third_party/benchmarks/supersede
  third_party/benchmarks/longmemeval
)
methods=(
  third_party/methods/pos
  third_party/methods/skill-state-runtime
  third_party/methods/scroll
  third_party/methods/vista
  third_party/methods/rlm
  third_party/methods/selfcompact
  third_party/methods/acm
  third_party/methods/belief-world-models
)
benchmarks=(
  third_party/benchmarks/loca-bench
  third_party/benchmarks/alfworld
  third_party/benchmarks/supersede
  third_party/benchmarks/longmemeval
)
references=(
  third_party/references/delayed-relevance
)
optional=(
  third_party/benchmarks/state-bench
  third_party/benchmarks/beam
)

group="${1:-core}"
case "$group" in
  core)       paths=("${core[@]}") ;;
  methods)    paths=("${methods[@]}") ;;
  benchmarks) paths=("${benchmarks[@]}") ;;
  references) paths=("${references[@]}") ;;
  optional)   paths=("${optional[@]}") ;;
  all)        paths=("${methods[@]}" "${benchmarks[@]}" "${references[@]}" "${optional[@]}") ;;
  *) echo "unknown group: $group (core|methods|benchmarks|references|optional|all)" >&2; exit 2 ;;
esac

git submodule update --init --depth 1 -- "${paths[@]}"
git submodule status -- "${paths[@]}"
