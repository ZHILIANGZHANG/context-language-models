# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""In-context skill evolution for CLM (see README.md in this directory)."""

from .propose import Proposal, Proposer, litellm_proposer, parse_proposal
from .tasks import Episode, HarborTaskSource, TaskSource, digest_trajectory, load_task_source

__all__ = [
    "Episode", "Evolution", "HarborTaskSource", "Proposal", "Proposer", "Score", "TaskSource",
    "digest_trajectory", "gate", "litellm_proposer", "load_task_source", "parse_proposal",
]


def __getattr__(name: str):
    # Imported lazily so that `python -m clm_icl.evolve` does not import evolve twice.
    if name in ("Evolution", "Score", "gate"):
        from . import evolve
        return getattr(evolve, name)
    raise AttributeError(name)
