# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The ``bash`` tool schema, the agent's whole action space.

Kept import-light (no litellm/harbor) so offline tooling can import the action space too.
"""

from __future__ import annotations

from typing import Any

BASH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Execute a bash command in a persistent shell. The working directory "
            "and environment variables are preserved between calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute.",
                }
            },
            "required": ["command"],
        },
    },
}
