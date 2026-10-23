"""
Pre-v2 reader modules, moved here day one of the refactor (REFACTORING.md §9).

Retained as the reference implementation while the v2 ports land; deleted per
sensor once its port is accepted. No re-exports — import from ``eoread.legacy``
explicitly.
"""

import sys
import types


def _install_jrc_stub() -> None:
    """Make the DIMAP readers importable without ``jrc_rayleigh_processor``.

    Four of the legacy DIMAP modules (geosat, planetscope, pleiades,
    pleiades_neo) do a top-level ``from jrc_rayleigh_processor.utils import
    iter_through_subset``. That package is private (not on PyPI or
    conda-forge), and the function is only ever used when a non-empty
    ``subset_meta_data`` list is passed — which no test does. Register a stub
    that raises if it is ever called, so a real installation (if present)
    always takes precedence.
    """
    try:
        import jrc_rayleigh_processor.utils  # noqa: F401
    except ImportError:
        module = types.ModuleType('jrc_rayleigh_processor')
        utils = types.ModuleType('jrc_rayleigh_processor.utils')

        def iter_through_subset(metadata, paths):
            raise NotImplementedError(
                'jrc_rayleigh_processor is not installed; legacy metadata '
                'subsetting (subset_meta_data) is unavailable')

        utils.iter_through_subset = iter_through_subset
        module.utils = utils
        sys.modules.setdefault('jrc_rayleigh_processor', module)
        sys.modules.setdefault('jrc_rayleigh_processor.utils', utils)


_install_jrc_stub()
