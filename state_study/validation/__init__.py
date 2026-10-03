"""Code for the 2026-10-03 validation round (see docs/validation_2026-10-03.md).

Generators (vdepth, retro, echo, ig) and their pilot build/score scripts are written from scratch.
The dr_* scripts replay traces recorded by the delayed-relevance project: they import that
project's environment from third_party/references/delayed-relevance at run time and never copy
its code or data into this repository.
"""
