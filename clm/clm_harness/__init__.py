# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""CLM harness: Harbor agent building blocks for context-managing agents.

Packages:
* ``clm_harness.clm_agent``               - ``ClmAgent``, the bash agent that manages its own
                                            context by editing a mirrored transcript file.
* ``clm_harness.context_env``             - writable-context environment: mirror sync, the
                                            fit/shrink edit gate, per-turn readout.
* ``clm_harness.context_utils``           - render the message log to an editable string and
                                            map an edited string back to messages.
* ``clm_harness.utils``                   - harness-agnostic infrastructure (token counting,
                                            snapshots, the bash tool schema, the sandbox).
* ``clm_harness.agent_trajectory_format`` - ATIF-CTX trajectory models, builder, validator.
* ``clm_harness.flops_metrics``           - KV-cache-aware inference-FLOPs metrics.
* ``clm_harness.task_templates``          - task-instruction (instance) prompt templates.

Kept intentionally light (no submodule imports here) so importing a leaf module
never drags in the whole serving stack.
"""
