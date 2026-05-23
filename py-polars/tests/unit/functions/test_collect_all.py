import pytest

import polars as pl


def test_collect_all_non_lazyframe_error() -> None:
    lf = pl.LazyFrame({"x": [1]})
    df = pl.DataFrame({"x": [2]})

    with pytest.raises(
        TypeError,
        match="in `pl.collect_all\\(\\)`, all elements must be LazyFrame instances; element 2 has type 'DataFrame'",
    ):
        pl.collect_all([lf, df])
