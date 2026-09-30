#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks of the SCR planning helpers (no GPU, no SGLang needed).

    python tests/unit_test.py
"""
import os
import sys

os.environ["KVREUSE_ENABLED"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from suffix_cache_reuse import overlay as o  # noqa: E402


class Req:
    pass


def test_find_session_respects_cache_key():
    ids = list(range(3000))
    sessions = {
        "a": {"sid": "a", "key": "tenant-a", "token_ids": ids, "side_slot": 0},
        "b": {"sid": "b", "key": None, "token_ids": ids, "side_slot": 1},
    }
    new = ids[:2000] + list(range(5000, 5100)) + ids[2100:]
    assert o._find_session(sessions, new, "tenant-a") == "a"
    assert o._find_session(sessions, new, None) == "b"
    assert o._find_session(sessions, new, "tenant-c") is None


def test_fallback_keeps_own_state_before_first_splice():
    r = Req()
    r._kvreuse_ssm_shadow = None
    o._v6_fallback_ssm(r, {"radix_a": True, "bi": 1, "ssm_restore": True, "side_slot": 3})
    assert r._kvreuse_ssm_shadow is None


def test_fallback_restores_after_materialised_lead():
    r = Req()
    r._kvreuse_ssm_shadow = None
    o._v6_fallback_ssm(r, {"ssm_restore": True, "side_slot": 3})
    assert r._kvreuse_ssm_shadow == 3


def test_fallback_without_restore_leaves_state():
    r = Req()
    r._kvreuse_ssm_shadow = None
    o._v6_fallback_ssm(r, {"radix_a": True, "bi": 2, "ssm_restore": False, "side_slot": 3})
    assert r._kvreuse_ssm_shadow is None


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"unit: {len(tests)} passed")
