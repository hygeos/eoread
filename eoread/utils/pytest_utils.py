#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def parametrize_dict(d) -> dict:
    """
    Allow passing args and ids to `pytest.mark.parametrize` with a dict syntax

    @pytest.mark.parametrize('a',  **parametrize_dict({
        'first case': 1,
        'second case': 2,
    }))   # equivalent to parametrize('a', [1, 2], ids=['first_case', 'second case'])
    def test(a):
        pass
    """
    return {"argvalues": list(d.values()), "ids": list(d.keys())}
