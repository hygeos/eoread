from core.tests.conftest import *

from pathlib import Path

import pytest

from eoread.reader import Reader
from eoread.samples import resolve


def get_sample(reader_or_id, product: str | None = None) -> Path:
    """Resolve a sample product, skipping the test when it is unavailable.

    The preferred form takes a reader class and uses its ``get_sample``
    classmethod::

        get_sample(Level1C_MSI)
        get_sample(Level1_OLI, 'OLI_L1TP_L9')

    A sample-product ID (registered in :mod:`eoread.samples`) is also
    accepted directly::

        get_sample('MSI_L1C')

    Tests run on real products only; missing data must produce a clean skip,
    never an error cascade.
    """
    try:
        if isinstance(reader_or_id, type) and issubclass(reader_or_id, Reader):
            return reader_or_id.get_sample(product)
        if product is not None:
            raise TypeError('product= is only valid with a Reader subclass')
        return resolve(reader_or_id)
    except (FileNotFoundError, KeyError) as exc:
        pytest.skip(str(exc))
