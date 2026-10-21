#!/usr/bin/env python3
"""
Base class for the v2 readers (see REFACTORING.md, §4).

A reader is a class — one per (sensor, level), grouped by sensor base.
Calling it reads the product and returns an instance holding the lazy dataset
in `.ds`:

    >>> from eoread.readers.msi import Level1C_MSI        # doctest: +SKIP
    >>> reader = Level1C_MSI('S2A_MSIL1C_...SAFE')        # doctest: +SKIP
    >>> ds = reader.ds                                    # doctest: +SKIP

The base signature is fixed for every reader; reader-specific options may be
added as extra keyword arguments with defaults (e.g. `resolution` on MSI):

    Reader(path, chunks=None, metadata_template=None, verbose=False)

Reading must stay lazy (dask-backed): nothing may be computed in `__init__`
beyond what is unavoidable to build the graph.
"""

from pathlib import Path

import xarray as xr

from .samples import resolve as resolve_sample
from .schema import SCHEMA_VERSION, validate
from .tools import filter_metadata, format_chunks


class ProductError(ValueError):
    """Base class for errors raised when a product cannot be read."""


class UnsupportedProductType(ProductError):
    """Raised when a product type/variant is not recognized (loud, never silent)."""


class UnsupportedProductVersion(ProductError):
    """Raised when a product version/baseline is outside the supported range."""


class Reader:
    """Base class for all v2 readers.

    Class attributes declaring the reader contract (enforced by
    `eoread.schema.validate` in `finalize`):

    sensor : str
        Sensor name, e.g. 'MSI'.
    level : str
        Product level, e.g. 'L1C'.
    chunks_default : int or tuple
        Default chunk size; per-sensor value carried over from the legacy reader.
    payload_var : str
        Primary payload variable ('Ltoa', 'Rtoa' or 'BT').
    required_vars : tuple of str
        Variables this reader must emit.
    sample_products : tuple of str
        Sample-product IDs registered in `eoread.samples` (e.g. 'MSI_L1C').
        The first resolvable product is returned by `get_sample`.
    VARIANTS : dict, optional
        Variant table for product types within a level (see `detect_variant`).
    flags : dict or None
        Flag definitions {name: bit}, attached as the `_flag_reader` attribute.

    Classes named with a leading underscore are sensor bases (shared
    implementation across levels), not public readers; `discover_readers()`
    skips them.
    """

    sensor = None
    level = None
    chunks_default = 500
    payload_var = None
    required_vars = ()
    sample_products = ()
    VARIANTS = {}
    flags = None

    def __init__(self, path, chunks=None, metadata_template=None, verbose=False):
        self.path = Path(path)
        self.chunks = format_chunks(chunks if chunks is not None else self.chunks_default)
        self.metadata_template = metadata_template
        self.verbose = verbose
        ds = self.read(self.path)
        self.ds = self.finalize(ds)

    # ------------------------------------------------------------------ api

    def read(self, path: Path) -> xr.Dataset:
        """Read the product and return a lazy dataset. Must be implemented."""
        raise NotImplementedError

    def finalize(self, ds: xr.Dataset, metadata: dict | None = None) -> xr.Dataset:
        """Attach curated metadata and enforce the canonical schema.

        Every reader ends in this method. It applies `metadata_template` to
        `metadata` when both are given, attaches the flag definitions, stamps
        `schema_version` and runs `eoread.schema.validate`. Readers with several
        metadata sources should merge them into one dict before calling.
        """
        if metadata is not None and self.metadata_template is not None:
            ds.attrs.update(filter_metadata(metadata, self.metadata_template))
        if self.flags is not None:
            ds.attrs['_flag_reader'] = self.flags
        ds.attrs['schema_version'] = SCHEMA_VERSION
        validate(ds, primary=self.payload_var, required_vars=self.required_vars)
        return ds

    # ------------------------------------------------------------- samples

    @classmethod
    def get_sample(cls, product: str | None = None) -> Path:
        """Resolve one of the reader's sample products to an existing path.

        With no argument, the IDs in `sample_products` are tried in order and
        the first one whose environment variable points to an existing path is
        returned.  Passing an explicit `product` ID selects it directly.

        Sample metadata (environment variable, description) lives in
        `eoread.samples`; readers only declare which IDs they support.
        """
        if product is not None:
            if product not in cls.sample_products:
                raise KeyError(
                    product+' is not part of declared sample products'
                    f'{cls.sample_products}')
            return resolve_sample(product)

        products = cls.sample_products
        if isinstance(products, str):
            products = (products,)
        if not products:
            raise NotImplementedError(
                f'{cls.__name__} does not declare any sample product '
                f'(set the sample_products class attribute)')

        errors = []
        for name in products:
            try:
                return resolve_sample(name)
            except (FileNotFoundError, KeyError) as exc:
                errors.append(f'  - {exc}')

        raise FileNotFoundError(
            f'no sample product available for {cls.sensor} {cls.level}:\n'
            + '\n'.join(errors))

    # ------------------------------------------------------------- variants

    @staticmethod
    def _lookup(header: dict, field):
        """Walk `field` (a key or a tuple of nested keys) into `header`."""
        if isinstance(field, str):
            field = (field,)
        node = header
        for key in field:
            if not isinstance(node, dict) or key not in node:
                raise KeyError(key)
            node = node[key]
        return node

    def detect_variant(self, header: dict) -> str:
        """Match product metadata against `VARIANTS`; loud raise on unknown type.

        Each variant entry must carry a `detect` mapping with `field` (a key or
        tuple of nested keys into `header`) and `value`. The match is exact by
        default; with `'prefix': True`, `value` is treated as a prefix of the
        string form of the field (v002 ECOSTRESS products append a scene
        suffix to their ShortName). Returns the variant name; raises
        `UnsupportedProductType` when none matches.
        """
        for vname, spec in self.VARIANTS.items():
            try:
                value = self._lookup(header, spec['detect']['field'])
            except KeyError:
                continue
            if spec['detect'].get('prefix'):
                if str(value).startswith(spec['detect']['value']):
                    return vname
            elif value == spec['detect']['value']:
                return vname
        known = ', '.join(self.VARIANTS) or '(none declared)'
        raise UnsupportedProductType(
            f'{self.sensor} {self.level}: unknown product type (known variants: {known})'
        )


def discover_readers() -> list[type[Reader]]:
    """
    Return every public reader class defined in `eoread.readers`.

    Walks the package, imports each module and collects the concrete
    `Reader` subclasses. Sensor bases (leading underscore) are skipped; they
    share implementation across levels and are not instantiated directly.
    Used by the conformance test and tooling.
    """
    import importlib
    import pkgutil

    import eoread.readers as readers_pkg

    found = []
    for module_info in pkgutil.iter_modules(readers_pkg.__path__):
        module = importlib.import_module(f'eoread.readers.{module_info.name}')
        for obj in vars(module).values():
            if (isinstance(obj, type) and issubclass(obj, Reader) and obj is not Reader
                    and not obj.__name__.startswith('_')
                    and obj.__module__ == module.__name__):
                found.append(obj)
    return sorted(found, key=lambda cls: (cls.sensor or '', cls.level or ''))