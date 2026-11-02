"""Import stand-in for ``jrc_rayleigh_processor``.

The v1 (0.5.0) DIMAP readers (geosat, pleiades, planetscope) import
``jrc_rayleigh_processor.utils.iter_through_subset`` at module level, but the
package is not a declared dependency of v1.  The benchmark only calls those
readers with path-only defaults (``meta_data=False``, ``subset_meta_data=[]``),
which never reaches that code path; this stub exists purely to satisfy the
import and fails loudly if it is ever actually used.
"""
