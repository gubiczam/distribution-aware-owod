"""Distribution-aware active annotation for Open-World Object Detection.

**This package deliberately re-exports nothing.** Import the module you need:

    from daowod import scoring, components      # equation (1)
    from daowod import annotation, study        # Contribution A
    from daowod import memory                   # Contribution B

The previous version eagerly imported seven modules here, which meant any
`from daowod import x` pulled in the legacy scorer, the synthetic diagnostics and
the image-level campaign. Every usage analysis therefore reported that dead code
as live, and the two research programs could never be separated. Keeping this file
empty is what makes the module boundaries real rather than nominal.

See `docs/research_design.md` for the architecture contract.
"""

__all__: list[str] = []
