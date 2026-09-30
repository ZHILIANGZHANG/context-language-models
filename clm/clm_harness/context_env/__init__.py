# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Writable-context environment: the conversation's env, not Harbor's VM.

``ClmAgent`` is the LLM loop. This package is everything ``step(command)``
hits — mirror sync, the fit/shrink gate and the per-turn readout.

Import the leaves; this file stays import-light::

    from clm_harness.context_env.env import ContextEnv
"""
