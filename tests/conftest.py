#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Allows to insert images in pytest html reports

- Requires pytest-html

- In your test, write your image in a ByteIO buffer and
  store the image content in request.session
    import io
    from .conftest import add_image_to_report
    from matplotlib import pyplot as plt
    def test_with_image(request):
        plt.plot(...)
        fp = io.BytesIO()
        plt.savefig(fp)
        add_image_to_report(request, fp)

    or:
    def test_with_image(request):
        plt.plot(...)
        conftest.savefig(request)


- Run pytest with the following options (can be added to pytest.ini):
    --html=tests/test_report.html --self-contained-html

- With this file conftest.py, the images will be added to the html report.
"""

import base64
import pytest


def add_image_to_report(request, fp):
    """
    Appends image data to request.node.images

    request: pytest `request` fixture

    fp: BytesIO
    """
    if not hasattr(request.node, 'images'):
        request.node.images = []
    fp.seek(0)
    data = fp.read()
    request.node.images.insert(0, data)


def savefig(request, **kwargs):
    """
    Wraps matplotlib's savefig to add image data to request.node.images

    `kwargs` are passed to `plt.savefig`
    """
    from matplotlib import pyplot as plt
    import io
    fp = io.BytesIO()
    plt.savefig(fp, **kwargs)
    add_image_to_report(request, fp)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item):
    pytest_html = item.config.pluginmanager.getplugin('html')
    outcome = yield
    report = outcome.get_result()
    extra = getattr(report, 'extra', [])
    if ((report.when == 'call')
            and (pytest_html is not None)):
        # add docstring
        doc = item.function.__doc__
        if doc is not None:
            extra.append(pytest_html.extras.html(f'<pre>{doc}</pre>'))

        # add images
        for image in getattr(item, 'images', []):
            b64data = base64.b64encode(image).decode('ascii')
            extra.append(pytest_html.extras.image(b64data))
        report.extra = extra
