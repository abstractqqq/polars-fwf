import polars as pl
import pyarrow as pa
import pytest

import ffwf as fw


def test_validate_specs_pl():
    df = pl.DataFrame({"a": [1, 1000], "b": ["abc", "defgh"]})
    specs = [
        fw.FieldSpec("a", 0, 2, "int"),  # 1000 is 4 chars, exceeds 2
        fw.FieldSpec("b", 2, 3, "str"),  # "defgh" is 5 chars, exceeds 3
    ]

    violations = fw.validate_specs_pl(df, specs)
    assert len(violations) == 2
    assert "Column 'a' has data longer (4) than specified length (2)" in violations[0]
    assert "Column 'b' has data longer (5) than specified length (3)" in violations[1]

    # Valid case
    specs_ok = [
        fw.FieldSpec("a", 0, 4, "int"),
        fw.FieldSpec("b", 4, 5, "str"),
    ]
    violations_ok = fw.validate_specs_pl(df, specs_ok)
    assert len(violations_ok) == 0


def test_validate_specs_arrow():
    table = pa.Table.from_pydict({"a": [1, 1000], "b": ["abc", "defgh"]})
    specs = [
        fw.FieldSpec("a", 0, 2, "int"),
        fw.FieldSpec("b", 2, 3, "str"),
    ]

    violations = fw.validate_specs_arrow(table, specs)
    assert len(violations) == 2
    assert "Column 'a' has data longer (4) than specified length (2)" in violations[0]
    assert "Column 'b' has data longer (5) than specified length (3)" in violations[1]


def test_validate_specs_newlines():
    # 1. Polars
    df = pl.DataFrame({"a": ["line 1\nline 2", "carriage\r"]})
    specs = [fw.FieldSpec("a", 0, 20, "str")]
    violations = fw.validate_specs_pl(df, specs)
    assert len(violations) == 1
    assert "contains 2 rows with line breaks" in violations[0]

    # 2. Arrow
    table = pa.Table.from_pydict({"a": ["line 1\nline 2", "carriage\r"]})
    violations_arrow = fw.validate_specs_arrow(table, specs)
    assert len(violations_arrow) == 1
    # Arrow count is total occurrences, not rows
    assert "contains 2 line break characters" in violations_arrow[0]
