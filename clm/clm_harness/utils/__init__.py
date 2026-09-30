# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Harness-agnostic utilities reusable by any clm_harness harness.

Modules (import the leaf directly; this package file stays import-light):
* ``tokens``       — tiktoken counting, per-turn snapshots, head/tail truncation.
* ``tool_schemas`` — the generic ``bash`` OpenAI tool schema.
"""
