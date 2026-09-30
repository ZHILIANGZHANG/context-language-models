# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Agent "skills" (e.g. the deep_research web tools) for the clm_harness harnesses.

Stock Harbor has no native ``skills:`` support, so a harness mounts them itself:
upload each skill dir into the sandbox, put its executables on PATH, export the API
keys they need, and inject the skill's ``SKILL.md`` into the system prompt.

Usage in a harness::

    self._skills = SkillMount(skill_dirs, skill_env_keys, skill_mount_dir)
    # in setup(), after creating _STATE_DIR:
    if self._skills.enabled:
        await self._skills.install(environment, state_dir=_STATE_DIR,
                                   persistent_bash=self.persistent_bash)
    # system prompt: base_template + self._skills.prompt_suffix()
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import Any

_DEFAULT_MOUNT = "/tmp/harbor_skills"
# A skill self-declares the env keys it needs in this optional manifest file (one per
# line or comma/space separated), so the HARNESS stays task-agnostic (it never names
# SERPER/JINA/etc.). Values are pulled from the process env (.env) at mount time.
_ENV_KEYS_MANIFEST = "skill_env_keys"


def _split(s: str) -> list[str]:
    return [p for p in re.split(r"[,\s]+", s.strip()) if p]


def _parse_skill_dirs(skill_dirs: str | list[str] | None) -> list[Path]:
    if not skill_dirs:
        return []
    parts = _split(skill_dirs) if isinstance(skill_dirs, str) else list(skill_dirs)
    out: list[Path] = []
    for p in parts:
        if not p:
            continue
        path = Path(p).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"skill dir not found: {path}")
        out.append(path)
    return out


class SkillMount:
    """Parsed skill dirs + env keys, with sandbox-install and prompt-injection.

    Env keys are resolved per-skill from each skill dir's ``skill_env_keys`` manifest
    (task-agnostic: the harness names no keys). ``skill_env_keys`` may be passed to
    OVERRIDE the manifests (e.g. for a skill without one). Key VALUES come from the
    process environment (populated from ``.env``); missing keys are silently skipped."""

    def __init__(
        self,
        skill_dirs: str | list[str] | None = None,
        skill_env_keys: str | list[str] | None = None,
        skill_mount_dir: str = _DEFAULT_MOUNT,
    ) -> None:
        self.mount_dir = skill_mount_dir.rstrip("/") or _DEFAULT_MOUNT
        self.dirs = _parse_skill_dirs(skill_dirs)
        keys = self._resolve_keys(skill_env_keys)
        self.env = {k: os.environ[k] for k in keys if k and os.environ.get(k)}

    def _resolve_keys(self, override: str | list[str] | None) -> list[str]:
        """Explicit override wins; otherwise union each skill's ``skill_env_keys``
        manifest. Order-preserving + de-duplicated."""
        if override:
            keys = _split(override) if isinstance(override, str) else list(override)
        else:
            keys = []
            for d in self.dirs:
                mf = d / _ENV_KEYS_MANIFEST
                if mf.is_file():
                    keys += _split(mf.read_text(encoding="utf-8"))
        return list(dict.fromkeys(keys))

    @property
    def enabled(self) -> bool:
        return bool(self.dirs)

    def _mount_for(self, skill_dir: Path) -> str:
        return f"{self.mount_dir}/{skill_dir.name}"

    def prompt_suffix(self) -> str:
        """The SKILL.md text (per skill) to append to the system prompt, telling the
        model the tools are on PATH at their mount dir."""
        parts: list[str] = []
        for d in self.dirs:
            skill_md = d / "SKILL.md"
            if not skill_md.is_file():
                continue
            mount = self._mount_for(d)
            parts.append(
                f"A skill you may use. Its tools are installed at `{mount}` and are already "
                f"on your PATH, so you can call them directly as shell commands (e.g. "
                f'`web_search "query"`). Ignore any different path mentioned inside the skill '
                f"text below; use `{mount}`.\n\n" + skill_md.read_text(encoding="utf-8")
            )
        return "\n\n---\n\n".join(parts)

    async def install(self, environment: Any, *, state_dir: str, persistent_bash: bool) -> None:
        """Upload each skill dir into the sandbox, put its executables on PATH, and
        export the (non-URL) API keys so every wrapped command sees them."""
        if not self.dirs:
            return
        exports: list[str] = []
        await environment.exec(command=f"mkdir -p {shlex.quote(self.mount_dir)}", timeout_sec=30)
        for d in self.dirs:
            mount = self._mount_for(d)
            await environment.upload_dir(str(d), mount)
            await environment.exec(
                command=f"chmod -R +x {shlex.quote(mount)} 2>/dev/null || true", timeout_sec=30
            )
            exports.append(f'export PATH={shlex.quote(mount)}:"$PATH"')
        for k, v in self.env.items():
            # URL-valued keys may be reverse-tunneled per-exec by the sandbox runtime; baking
            # the raw URL into the sourced env file would shadow that rewrite, so skip.
            if v.startswith(("http://", "https://")):
                continue
            exports.append(f"export {k}={shlex.quote(v)}")
        if persistent_bash and exports:
            block = "\n".join(exports)
            await environment.exec(
                command=(
                    f"mkdir -p {state_dir} && cat >> {state_dir}/env <<'HARBOR_SKILL_ENV'\n"
                    f"{block}\nHARBOR_SKILL_ENV"
                ),
                timeout_sec=10,
            )
