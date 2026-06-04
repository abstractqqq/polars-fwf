import datetime
import os

import polars as pl
import pytest

import ffwf as fw


def test_write_fwf_basic_pl(tmp_path):
    path = str(tmp_path / "test.fwf")
    df = pl.DataFrame(
        {
            "id": [1, 2, 100],
            "name": ["Alice", "Bob", "Charlie"],
            "val": [1.1, 2.22, 3.333],
            "active": [True, False, None],
        }
    )

    # Test inference
    specs_dict = fw.write_fwf_pl(df, path)

    assert "id" in specs_dict
    assert "name" in specs_dict
    assert "val" in specs_dict
    assert "active" in specs_dict

    # Read back to verify
    specs = [
        fw.FieldSpec(
            "id", specs_dict["id"]["offset"], specs_dict["id"]["length"], "int"
        ),
        fw.FieldSpec(
            "name", specs_dict["name"]["offset"], specs_dict["name"]["length"], "str"
        ),
        fw.FieldSpec(
            "val", specs_dict["val"]["offset"], specs_dict["val"]["length"], "f64"
        ),
        fw.FieldSpec(
            "active",
            specs_dict["active"]["offset"],
            specs_dict["active"]["length"],
            "str",
        ),
    ]

    df_read = fw.read_fwf_pl(path, specs)
    assert df_read.shape == (3, 4)
    assert df_read["id"].to_list() == [1, 2, 100]
    assert [s.strip() for s in df_read["name"]] == ["Alice", "Bob", "Charlie"]


def test_write_fwf_specs_pl(tmp_path):
    path = str(tmp_path / "test_spec.fwf")
    df = pl.DataFrame({"a": [1, 10], "b": ["x", "yz"]})

    specs = [fw.FieldSpec("a", 0, 5, "int"), fw.FieldSpec("b", 5, 5, "str")]

    fw.write_fwf_pl(df, path, specs=specs)

    with open(path, "rb") as f:
        lines = f.readlines()
        assert lines[0] == b"    1x    \n"
        assert lines[1] == b"   10yz   \n"


def test_write_fwf_truncation_pl(tmp_path):
    path = str(tmp_path / "fail.fwf")
    df = pl.DataFrame({"a": [1000]})
    # Spec only allows 2 chars, but 1000 needs 4. Native writer truncates.
    specs = [fw.FieldSpec("a", 0, 2, "int")]

    fw.write_fwf_pl(df, path, specs=specs)

    with open(path, "rb") as f:
        content = f.read()
        # "1000" truncated to 2 chars -> "10"
        assert content == b"10\n"


def test_write_fwf_bool_treatment_pl(tmp_path):
    path = str(tmp_path / "bool.fwf")
    df = pl.DataFrame({"a": [True, False, None]})
    # Width 3 for YES/NO/---
    specs = [fw.FieldSpec("a", 0, 3, "str")]
    fw.write_fwf_pl(df, path, specs=specs, bool_treatment=("YES", "NO ", "---"))

    with open(path, "rb") as f:
        lines = f.readlines()
        assert lines[0] == b"YES\n"
        assert lines[1] == b"NO \n"
        assert lines[2] == b"---\n"


def test_sink_fwf_pl(tmp_path):
    path = str(tmp_path / "sink.fwf")
    lf = pl.DataFrame({"a": [1, 2]}).lazy()
    specs = [fw.FieldSpec("a", 0, 5, "int")]
    fw.sink_fwf_pl(lf, path, specs=specs)

    with open(path, "rb") as f:
        lines = f.readlines()
        assert lines[0] == b"    1\n"
        assert lines[1] == b"    2\n"


def test_write_fwf_large_floats_pl(tmp_path):
    path = str(tmp_path / "large_floats.fwf")
    # Huge float
    df = pl.DataFrame({"val": [1.23456789e300, 1.23456789e-10]})

    # Test with 2 significant digits (lexical behavior)
    specs_dict = fw.write_fwf_pl(df, path, decimals=2)

    with open(path, "rb") as f:
        lines = f.readlines()
        # lexical-core with 2 sig digits might produce 1.2e+300 or similar
        assert b"1.2" in lines[0]


def test_write_fwf_truncation_logic_pl(tmp_path):
    path = str(tmp_path / "trunc.fwf")
    # 1.999 with decimals=1 might be 2.0 or 2 depending on rounding/trimming
    df = pl.DataFrame({"val": [1.999]})
    specs = [fw.FieldSpec("val", 0, 5, "f64")]
    fw.write_fwf_pl(df, path, specs=specs, decimals=1)

    with open(path, "rb") as f:
        line = f.read().rstrip(b"\n")
        # lexical-core with trim_floats(true) will produce "2"
        assert b"2" in line


def test_write_fwf_negatives_pl(tmp_path):
    path = str(tmp_path / "negatives.fwf")
    df = pl.DataFrame({"i": [-1, -100], "f": [-1.23456, -0.0001]})

    # Specs with enough width for signs
    specs = [fw.FieldSpec("i", 0, 5, "int"), fw.FieldSpec("f", 5, 8, "float")]

    fw.write_fwf_pl(df, path, specs=specs, decimals=2)

    with open(path, "rb") as f:
        lines = f.readlines()
        # "-1" padded to width 5 -> "   -1"
        # "-1.23456" with decimals=2 -> "-1.2" (or similar depending on max_significant_digits)
        assert b"-1" in lines[0]
        assert b"-" in lines[0][5:]


def test_write_fwf_arrow(tmp_path):
    import pyarrow as pa

    path = str(tmp_path / "arrow.fwf")
    table = pa.Table.from_pydict({"a": [1, 2], "b": ["x", "y"]})
    specs = [fw.FieldSpec("a", 0, 5, "int"), fw.FieldSpec("b", 5, 5, "str")]

    fw.write_fwf_arrow(table, path, specs)

    with open(path, "rb") as f:
        lines = f.readlines()
        assert lines[0] == b"    1x    \n"
        assert lines[1] == b"    2y    \n"
