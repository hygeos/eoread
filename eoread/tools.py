import atexit

import dask.array as da
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
import numpy as np

from core.interpolate import Linear, Nearest
from core.import_utils import import_module
from core.uncompress import uncompress
from core.geo.naming import names
from core.tools import only
from core import env

from typing import Union, Literal
from xarray import DataArray, Dataset, open_dataarray
from tempfile import TemporaryDirectory
from pathlib import Path


def format_chunks(chunks: Union[int, list, tuple, dict]) -> dict:
    """
    Normalize chunk specification to a dictionary format.
    
    Converts various chunk size specifications (int, list, tuple) to a
    standardized dictionary with 'rows' and 'columns' keys.
    
    Parameters
    ----------
    chunks : int, list, tuple, or dict
        Chunk size specification:
        - int: Same chunk size for both dimensions
        - list/tuple: [rows_chunk, columns_chunk]
        - dict: Already in correct format
        
    Returns
    -------
    dict
        Dictionary with 'rows' and 'columns' chunk sizes.
        
    Raises
    ------
    AssertionError
        If chunks format is invalid.
        
    Examples
    --------
    >>> format_chunks(500)
    {'rows': 500, 'columns': 500}
    >>> format_chunks([1000, 500])
    {'rows': 1000, 'columns': 500}
    """
    
    # Manage different chunk types
    if isinstance(chunks, int):
        chunks = [chunks, chunks]
    if isinstance(chunks, list|tuple):
        assert len(chunks) == 2
        chunks = {str(names.rows): chunks[0], str(names.columns): chunks[1]}
    
    assert len(chunks) == 2, 'chunks should be for spatial dimensions'
    assert str(names.rows) in chunks and str(names.columns) in chunks
    return chunks


def filter_metadata(metadata: dict, template: list) -> dict:
    """
    Short method to filter metadata dictionary based on a template

    Parameters
    ----------
    metadata : dict
        Dictionary corresponding to metadata.
    template : list
        List of list describing nodes to keep.
    
    Examples
    --------
    >> d = {'a': 0, 'b': {'c': 1, 'd':2}}
    >> t = [['a'], ['b','c']]
    >> filter_metadata(d, t)
    {'a': 0, 'b': {'c': 1}}
    """  
    
    # Populate new dictionary with the selected nodes.  Intermediate nodes are
    # created once and reused, so several leaves under the same parent are all
    # kept (e.g. [['b', 'c'], ['b', 'd']] -> {'b': {'c': ..., 'd': ...}}).
    result = {}
    for key_list in template:
        current_meta = metadata
        current_result = result

        # Traverse the nested dictionary using the provided keys
        for key in key_list[:-1]:
            if not isinstance(current_meta, dict):
                raise ValueError(
                    f'conflicting template paths at {key!r}: a previous leaf '
                    f'is used as a group (template: {key_list})')
            if key not in current_meta:
                raise ValueError(f'{key} not found (template: {key_list})')

            child = current_result.get(key)
            if child is None:
                child = {}
                current_result[key] = child
            elif not isinstance(child, dict):
                raise ValueError(
                    f'conflicting template paths at {key!r}: a leaf and a '
                    f'group use the same key (template: {key_list})')

            current_result = child
            current_meta = current_meta[key]

        # If the last key is present, add it to the result
        leaf = key_list[-1]
        if not isinstance(current_meta, dict) or leaf not in current_meta:
            raise ValueError(f'Leaf {leaf} not found in metadata (template: {key_list})')
        current_result[leaf] = current_meta[leaf]

    return result


# ---------------------------------------------------------------------------
# Process-exit hygiene (shared by every rasterio-backed reader)
# ---------------------------------------------------------------------------

_EXIT_CLEANUP_INSTALLED = False


def _close_open_file_managers():
    """Close every still-open xarray CachingFileManager, at process exit.

    xarray's rasterio backend opens each file through a CachingFileManager and
    leaves it open so lazy windowed reads can happen later (JP2 via rioxarray,
    GeoTIFF natively). If such a manager survives to interpreter finalization,
    its ``__del__`` (and the eviction of xarray's global FILE_CACHE) tries to
    close the dataset after module state has been torn down; the resulting
    exception inside ``__del__`` is emitted by CPython >= 3.12 as nested 'Error
    in sys.excepthook' pairs with empty tracebacks — one per open file, after
    the program already exited cleanly (RC=0). Closing from atexit, while
    modules are still alive, leaves nothing for finalization to close. This
    handler must never raise: a failure here would print its own traceback to
    stderr at exit, recreating the very noise it removes.
    """
    try:
        import gc

        from xarray.backends.file_manager import CachingFileManager
    except Exception:
        return
    for obj in gc.get_objects():
        if type(obj) is CachingFileManager:
            try:
                obj.close(needs_lock=False)
            except Exception:
                pass


def install_exit_cleanup() -> None:
    """Register the atexit file-manager close, once per process."""
    global _EXIT_CLEANUP_INSTALLED
    if not _EXIT_CLEANUP_INSTALLED:
        atexit.register(_close_open_file_managers)
        _EXIT_CLEANUP_INSTALLED = True


def _split_blocks(n: int, chunk: int) -> list[tuple[int, int]]:
    """
    Split the range [0, n) into consecutive blocks of at most `chunk` elements.

    Returns a list of (start, size) tuples covering the range exactly once.
    """
    assert chunk >= 1, 'chunk must be a positive integer'
    return [(i, min(chunk, n - i)) for i in range(0, n, chunk)]


def _axis_window(indexer, positions: np.ndarray, is_linear: bool) -> tuple:
    """
    Locate a block of target positions in the source grid and derive the minimal
    source window it touches.

    Returns (lo, hi, i_lo, w_lo, i_hi, w_hi) where [lo, hi) is the window in
    source indices and the index arrays are already shifted to window-local
    coordinates. For nearest-neighbour selection there is a single index and no
    weights (i_hi/w_hi are None).
    """
    res = indexer(positions)
    if is_linear:
        i_lo, w_lo = res[0]
        i_hi, w_hi = res[1]
        lo = int(min(i_lo.min(), i_hi.min()))
        hi = int(max(i_lo.max(), i_hi.max())) + 1
        return lo, hi, i_lo - lo, w_lo, i_hi - lo, w_hi
    (i_lo, _), = res
    lo = int(i_lo.min())
    hi = int(i_lo.max()) + 1
    return lo, hi, i_lo - lo, None, None, None


def _interp_block(win: np.ndarray, i0a, w0a, i0b, w0b, i1a, w1a, i1b, w1b) -> np.ndarray:
    """
    Interpolate one output block from a source window.

    `win` is the 2D numpy source window; per-axis index/weight arrays are 1D and
    shifted to window-local coordinates (i*_b/w*_b are None for nearest
    neighbour). The expression mirrors core.interpolate.interp_var term by term,
    so results are bit-for-bit identical to a full-grid interpolation.
    """
    if w0a is None:  # nearest neighbour: single index, unit weight
        return win[i0a[:, None], i1a[None, :]] * 1
    r = win[i0a[:, None], i1a[None, :]] * (w0a[:, None] * w1a[None, :])
    r += win[i0a[:, None], i1b[None, :]] * (w0a[:, None] * w1b[None, :])
    r += win[i0b[:, None], i1a[None, :]] * (w0b[:, None] * w1a[None, :])
    r += win[i0b[:, None], i1b[None, :]] * (w0b[:, None] * w1b[None, :])
    return r


def _blockwise_interp(src, src_coords: tuple, targets: tuple, chunks: tuple,
                      method: Literal['linear', 'nearest'] = 'linear'):
    """
    Lazily resample a 2D field onto a target grid spanning the source range.

    Returns a dask array of shape (len(targets[0]), len(targets[1])) whose
    element [j, i] is `src` interpolated at position targets[0][j] along axis 0
    and targets[1][i] along axis 1, using the same core.interpolate indexers as
    interp. The work is split into one dask task per output block; each task
    only touches the source window its target positions fall in (the target grid
    spans exactly the source range, so no extrapolation happens).

    Results are bit-for-bit identical to a full-grid core.interpolate.interp
    evaluation (same indexers, same per-point arithmetic), including the NaN*0
    propagation of zero-weight neighbour terms.

    Parameters
    ----------
    src : dask array or numpy array
        2D source field.
    src_coords : tuple of two 1D arrays
        Source coordinate values for each axis (monotonic).
    targets : tuple of two 1D arrays
        Target positions in source-coordinate units, within the source range.
    chunks : tuple of two ints
        Output chunk size per axis.
    method : 'linear' or 'nearest'
        Interpolation mode ('nearest' corresponds to Nearest / repeat).

    Returns
    -------
    dask.array.Array
        Lazily resampled field, one task per output block.
    """
    is_linear = method == 'linear'
    proxy = Linear if is_linear else Nearest
    # The indexer only depends on the source coordinates: get_indexer ignores
    # the variable passed to __init__, so a None placeholder is sufficient.
    indexers = [proxy(None).get_indexer(DataArray(c)) for c in src_coords]

    row_blocks = _split_blocks(len(targets[0]), chunks[0])
    col_blocks = _split_blocks(len(targets[1]), chunks[1])
    win0 = [_axis_window(indexers[0], targets[0][r0:r0 + nr], is_linear) for r0, nr in row_blocks]
    win1 = [_axis_window(indexers[1], targets[1][c0:c0 + nc], is_linear) for c0, nc in col_blocks]

    # Result dtype: same expressions as _interp_block (weights are float64 for
    # linear; unit int weights for nearest).
    probe = np.zeros(1, dtype=src.dtype)
    dtype = (probe * (np.float64(0.5) * np.float64(0.5))).dtype if is_linear else (probe * 1).dtype

    name = 'eoread-blockwise-interp-' + tokenize(src, src_coords, targets, chunks, method)
    dsk = {}
    deps = []  # dask arrays; their layers are merged into the result graph
    for a, ((r0, nr), w_a) in enumerate(zip(row_blocks, win0)):
        lo0, hi0, i0a, w0a, i0b, w0b = w_a
        for b, ((c0, nc), w_b) in enumerate(zip(col_blocks, win1)):
            lo1, hi1, i1a, w1a, i1b, w1b = w_b
            if isinstance(src, da.Array):
                window = src[lo0:hi0, lo1:hi1]
                if window.numblocks != (1, 1):
                    window = window.rechunk((hi0 - lo0, hi1 - lo1))
                # Single-block array: its own layer holds the one output key.
                (win_key,) = window.__dask_graph__().layers[window.name]
                deps.append(window)
            else:
                win_key = src[lo0:hi0, lo1:hi1]
            dsk[(name, a, b)] = (_interp_block, win_key, i0a, w0a, i0b, w0b, i1a, w1a, i1b, w1b)

    hlg = HighLevelGraph.from_collections(name, dsk, dependencies=deps)
    return da.Array(
        hlg, name,
        chunks=(tuple(nr for _, nr in row_blocks), tuple(nc for _, nc in col_blocks)),
        dtype=dtype,
    )


def _upsample(array: DataArray, output_shape: dict, chunks: dict, method: str) -> DataArray:
    """
    Lazy over-sampling branch of spatial_resample.

    The target grid is affine (linspace between first and last source
    coordinates), so every output block maps to a contiguous source window and
    no pixel data is read at construction time. Results are bit-for-bit
    identical to the former compute()-based implementation, including its output
    convention: dims (dims[1], dims[0]), i.e. the correctly placed grid with
    the axis order reversed (e.g. ('x', 'y') for a ('y', 'x') input) — callers
    that want the source dim order transpose the result back.

    Parameters
    ----------
    array : DataArray
        2D source field with index coordinates on both dims.
    output_shape : dict
        Target size per dim.
    chunks : dict
        Output chunk size per dim.
    method : 'linear' or 'repeat'
        Interpolation mode.

    Returns
    -------
    DataArray
        Lazily resampled field with the historical (dims[1], dims[0]) layout.
    """
    dims = array.dims
    c0 = np.asarray(array.coords[dims[0]].values)
    c1 = np.asarray(array.coords[dims[1]].values)
    # The legacy branch built its target grids with dask.array.linspace, whose
    # endpoint is (num - 1) * step accumulated in float — not always bit-equal
    # to np.linspace(stop). Reuse it so the indexer weights at the last
    # coordinate match exactly. Computing these 1D grids eagerly is cheap and
    # reads no pixel data.
    t0 = da.linspace(c0[0], c0[-1], output_shape[dims[0]]).compute(scheduler='sync')
    t1 = da.linspace(c1[0], c1[-1], output_shape[dims[1]]).compute(scheduler='sync')

    # out[j, i] holds the value at (axis-0 position = t0[j], axis-1 position =
    # t1[i]) — the correctly placed grid in source dim order. Transposing
    # yields the historical layout: element [a, b] of dims (dims[1], dims[0])
    # is f(t0[b], t1[a]).
    out = _blockwise_interp(
        array.data, (c0, c1), (t0, t1),
        (chunks[dims[0]], chunks[dims[1]]),
        'linear' if method == 'linear' else 'nearest')

    return DataArray(out.T, dims=(dims[1], dims[0]), attrs=array.attrs)


def spatial_resample(
        array: DataArray,
        output_shape: dict, 
        chunks: dict,
        method: Literal['linear','repeat'] = 'linear'
    ) -> DataArray:
    """
    Lazily resample a 2D DataArray onto a target shape.

    An exact-integer-ratio downsample averages the integer pixel groups; every
    other case (upsamples and non-integer-ratio downsamples) interpolates onto
    a target grid spanning the source coordinate range (see `_upsample`). The
    interpolation branch returns the result with swapped dim labels, so
    callers must `.transpose(rows, columns)` it.

    Parameters
    ----------
    array : DataArray
        2D input; the coordinates of both dims are the source grid.
    output_shape : dict {dim: int}
        Target size per dim (keys must be the input's dims).
    chunks : dict {dim: int}
        Output chunk sizes, keyed by dim (must match `output_shape`).
    method : 'linear' or 'repeat'
        Interpolation mode of the interpolation branch.

    Returns
    -------
    DataArray
        The resampled field on the target shape and chunks.
    """
    
    # Check inputs compliance
    array = array.squeeze()
    assert len(array.shape) == 2, 'Array should be 2D'
    assert all(k in array.dims for k in output_shape.keys())
    assert chunks.keys() == output_shape.keys()
    
    # Compute the ratio of transformation
    dims = array.dims
    ratio = {d: array.sizes[d]/output_shape[d] for d in dims}
    
    # No resolution change => return the input DataArray
    if all(v == 1 for v in ratio.values()):
        return array
    
    # Exact-integer-ratio downsample: average the integer pixel groups.
    if any(v > 1 for v in ratio.values()) and all(r == int(r) for r in ratio.values()):
        params = {d: int(ratio[d]) for d in dims}
        resampled = array.coarsen(**params, boundary="trim")
        resampled = resampled.mean().chunk(chunks)
        
    # Any other case (every upsample, plus non-integer-ratio downsamples):
    # lazy blockwise interpolation (see _upsample); no pixel data is read at
    # construction time. The OLI panchromatic -> 30 m resample lands here: the
    # two grids share the same geographic span and the pan grid is one pixel
    # short of exactly twice the reflective one, so the affine target positions
    # fall on exact half-pixel offsets of the source grid.
    else:
        resampled = _upsample(array, output_shape, chunks, method)

    return resampled


def open_raster(
        dirname: Union[str, Path], 
        pattern: str, 
        compress_ext: str = None, 
        engine='h5netcdf'
    ) -> DataArray:
    """
    Find and open a raster file matching a pattern.
    
    Searches for a single file matching the pattern and optionally
    decompresses it before opening.
    
    Parameters
    ----------
    dirname : str or Path
        Directory to search in.
    pattern : str
        Glob pattern to match (e.g., '*_B01.jp2').
    compress_ext : str, optional
        If provided, uncompresses file with this extension first.
        Default is None.
    engine : str, optional
        xarray engine for opening the file. Default is 'h5netcdf'.
        
    Returns
    -------
    DataArray
        DataArray with the raster data (squeezed to remove size-1 dimensions).
        
    Raises
    ------
    AssertionError
        If pattern matches zero or multiple files.
        
    Examples
    --------
    >>> arr = open_raster('/data/', '*_B02.tif', engine='rasterio')
    >>> arr = open_raster('/data/', '*_CLD.zip', compress_ext='.zip')
    """
    # Find file path
    path = only(list(Path(dirname).glob(pattern)))
    
    # Open and uncompress if needed 
    if compress_ext: 
        with TemporaryDirectory() as tmpdir:
            path = uncompress(path, tmpdir)
            return open_dataarray(path, engine=engine).squeeze()
    
    return open_dataarray(path, engine=engine).squeeze()


def collect_sample(
        variable: str,
        provider: Literal[None, 'cnes'],
        sand_collection: str|None = None,
        level: int|None = None
    ) -> Path:
    
    # Check if user has provided a path
    variable = env.getvar(variable, default='')  
    
    # If not provided, try to download a sample with SAND
    if variable == '':
        
        if provider is not None:
            assert sand_collection and level, 'Provide kwargs for SAND downloader'
        else: 
            raise ValueError('')
        
        # Check SAND importation
        try: 
            from sand.sample_product import products
        except ImportError:
            raise ImportError('To use get_sample function, you need to install SAND module')
        
        # Collect appropriated provider
        downloader = {
            'cdse': 'sand.copernicus_dataspace.DownloadCDSE',
            'cnes': 'sand.cnes.DownloadCNES',
            'eumdac': 'sand.eumdac.DownloadEumDAC',
            'nasa': 'sand.nasa.DownloadNASA',
            'usgs': 'sand.usgs.DownloadUSGS',
        }[provider]
        
        # Retrieve name of example product
        prod_id = products[sand_collection][f'l{level}_product']
        
        # Download product with SAND
        dl = import_module(downloader)()
        directory = env.getdir('DIR_SAMPLES')/sand_collection
        target = dl.download_file(prod_id, directory)
        
        assert target.exists()
        return target
        
    else:
        return Path(variable)
    

def crop(
        ds: Dataset, 
        latmin: float|None = None, 
        latmax: float|None = None,
        lonmin: float|None = None, 
        lonmax: float|None = None,
        drop: bool = True
    ) -> Dataset:
    """
    Crop output of eoread reader based on latitude and longitude arrays.

    Parameters
    ----------
    ds : xr.Dataset
        Output from an eoread reader.
    latmin : float or None, optional
        Minimum of latitude. Default is None.
    latmax : float or None, optional
        Maximum of latitude. Default is None.
    lonmin : float or None, optional
        Minimum of longitude. Default is None.
    lonmax : float or None, optional
        Maximum of longitude. Default is None.
    drop : bool, optional
        Option to drop invalid pixels. If False, invalid pixels are set to NaN. Default is True.
    """
    # Filter latitude and longitude
    latmask = (ds[str(names.lat)] >= latmin) & (ds[str(names.lat)] <= latmax)
    lonmask = (ds[str(names.lon)] >= lonmin) & (ds[str(names.lon)] <= lonmax)
    
    # Load masks to determine output shape 
    if drop: 
        latmask = latmask.compute()
        lonmask = lonmask.compute()
        
    return ds.where(latmask & lonmask, drop=drop)


def _unquote(value: str) -> str:
    """Strip one pair of surrounding double quotes from an HDF-EOS leaf value."""
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    return value


def parse_hdf4_metadata(text: str) -> dict:
    """
    Parse an HDF-EOS text metadata block into a nested dict.

    Handles both the padded layout (CoreMetadata, ArchiveMetadata:
    ``KEY<pad>=<pad>VALUE``, with leaf objects split over several lines) and
    the compact one (StructMetadata). ``GROUP``/``OBJECT`` open a nested dict,
    closed by ``END_GROUP``/``END_OBJECT``; any other ``KEY = VALUE`` line is a
    leaf stored as a string, with surrounding double quotes stripped (numeric
    values and list literals stay bare).

    Parameters
    ----------
    text : str
        Raw metadata text (e.g. the file's ``CoreMetadata.0`` attribute).

    Returns
    -------
    dict
        Nested tree; leaf objects map to their field values, e.g.
        ``{'RANGEBEGINNINGTIME': {'NUM_VAL': '1', 'VALUE': '17:40:00.000000'}}``.
    """
    lines = [line.strip() for line in text.splitlines()]

    def parse_block(i: int):
        node = {}
        while i < len(lines):
            line = lines[i]
            if not line:
                i += 1
                continue
            key, sep, value = line.partition('=')
            if not sep:
                # The text blocks end with a bare 'END' terminator.
                if line == 'END':
                    return node, i + 1
                raise ValueError(f'unparseable metadata line: {line!r}')
            key, value = key.strip(), value.strip()
            if key in ('GROUP', 'OBJECT'):
                child, i = parse_block(i + 1)
                node[value] = child
            elif key.startswith('END_'):
                return node, i + 1
            else:
                node[key] = _unquote(value)
                i += 1
        return node, i

    tree, _ = parse_block(0)
    return tree