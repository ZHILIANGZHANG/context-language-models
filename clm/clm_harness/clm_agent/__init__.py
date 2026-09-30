# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Live-context harness: context editing is just bash.

The action space is a single ``bash`` tool. The harness mirrors the model's
editable context to ``/tmp/.live_ctx/LIVE_CTX_MAIN.txt`` in the sandbox before each
command; the model compacts by editing that file with ordinary shell tools, and
the harness maps changes back into the conversation via
``clm_harness.context_utils.context_string``.

Because edits run inside the sandbox VM (not an in-process ``exec``), there is no
sandboxed-code-execution concern. Modules:
* ``harness``      — ``ClmAgent`` (LLM loop, bash parse, submit).
* ``prompts.yaml`` — system/instance prompts (the mirror-file protocol lives here).

Writable-context I/O, the fit/shrink gate and the per-turn readout live in
``clm_harness.context_env``.
"""
