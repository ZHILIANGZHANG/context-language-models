# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""``clm-harbor``: the Harbor CLI with CLM registered as ``-a clm-minimal``.

Every argument is passed to Harbor unchanged, except:

* ``-a clm-minimal`` / ``--agent clm-minimal`` selects ``ClmAgent``;
* ``--clm-config <bcp|edgebench|path.yaml>`` expands a config into ``--agent-kwarg``
  flags and exports its ``env`` entries;
* when the CLM agent is selected, ``context_budget_tokens=32000`` and
  ``cost_metric=usd`` are added unless given.

Example::

    clm-harbor run -p path/to/task -a clm-minimal -m openai/<model> \\
        --agent-kwarg api_base=http://localhost:8000/v1
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

AGENT_NAME = "clm-minimal"
IMPORT_PATH = "clm_harness.clm_agent.harness:ClmAgent"
CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_KWARGS = {"context_budget_tokens": "32000", "cost_metric": "usd"}


def _config_items(spec: str) -> tuple[dict[str, str], dict[str, str]]:
    import yaml

    path = CONFIG_DIR / f"{spec}.yaml" if (CONFIG_DIR / f"{spec}.yaml").is_file() else Path(spec)
    if not path.is_file():
        known = ", ".join(sorted(p.stem for p in CONFIG_DIR.glob("*.yaml")))
        raise SystemExit(f"clm-harbor: unknown config {spec!r} (known: {known}, or a YAML path)")
    cfg = yaml.safe_load(path.read_text()) or {}

    def fmt(v: object) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        return "none" if v is None else str(v)

    env = {str(k): fmt(v) for k, v in (cfg.get("env") or {}).items()}
    kwargs = {str(k): fmt(v) for k, v in (cfg.get("agent_kwargs") or {}).items()}
    return env, kwargs


def rewrite_args(argv: list[str]) -> list[str]:
    """Map the CLM agent name and config onto plain Harbor arguments."""
    out: list[str] = []
    config: str | None = None
    selected = False
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-a", "--agent") and i + 1 < len(argv) and argv[i + 1] == AGENT_NAME:
            out += [arg, IMPORT_PATH]
            selected = True
            i += 2
            continue
        if arg == f"--agent={AGENT_NAME}":
            out.append(f"--agent={IMPORT_PATH}")
            selected = True
            i += 1
            continue
        if arg == "--clm-config" and i + 1 < len(argv):
            config = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--clm-config="):
            config = arg.split("=", 1)[1]
            i += 1
            continue
        out.append(arg)
        i += 1

    if not selected:
        if config is not None:
            raise SystemExit("clm-harbor: --clm-config needs -a clm-minimal")
        return out

    given: set[str] = set()
    for j, arg in enumerate(out):
        if arg in ("--agent-kwarg", "--ak") and j + 1 < len(out):
            given.add(out[j + 1].split("=", 1)[0])
        elif arg.startswith(("--agent-kwarg=", "--ak=")):
            given.add(arg.split("=", 1)[1].split("=", 1)[0])

    extra: dict[str, str] = {}
    if config is not None:
        env, kwargs = _config_items(config)
        os.environ.update(env)
        extra.update({k: v for k, v in kwargs.items() if k not in given})
    for k, v in DEFAULT_KWARGS.items():
        if k not in given and k not in extra:
            extra[k] = v
    for k, v in extra.items():
        out += ["--agent-kwarg", f"{k}={v}"]
    return out


def main() -> None:
    from harbor.cli.main import app

    sys.argv = [sys.argv[0], *rewrite_args(sys.argv[1:])]
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    app()


if __name__ == "__main__":
    main()
