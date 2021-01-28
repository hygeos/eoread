from contextlib import contextmanager
from dask.diagnostics import Profiler, ResourceProfiler, visualize

@contextmanager
def DaskProfiler(filename='profile_dask.html'):
    """
    A simple wrapper for profiling tools in `dask.diagnostics`
    """
    with ResourceProfiler() as rprof,\
            Profiler() as prof:
        yield
        visualize([prof, rprof], filename)
