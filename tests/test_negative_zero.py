import math
import os

import polars as pl
import pytest

import ffwf as fw


def test_read_negative_zero_pl():
    specs = [fw.FieldSpec("val", 0, 5, "f64")]
    # Test file with negative zero
    path = "tests/data_neg_zero.fwf"
    os.makedirs("tests", exist_ok=True)
    with open(path, "wb") as f:
        f.write(b" -0.0\n")

    df = fw.read_fwf_pl(path, specs)
    val = df["val"][0]
    assert val == 0.0
    # math.copysign is used to detect negative zero
    assert math.copysign(1.0, val) == -1.0


def test_write_negative_zero_pl(tmp_path):
    path = str(tmp_path / "write_neg_zero.fwf")

    # Create DataFrame with negative zero
    df = pl.DataFrame({"f": [math.copysign(0.0, -1.0)]})
    assert math.copysign(1.0, df["f"][0]) == -1.0

    # Spec width 4 is enough for "-0.0"
    specs = [fw.FieldSpec("f", 0, 4, "float")]

    fw.write_fwf_pl(df, path, specs=specs, decimals=1)

    with open(path, "rb") as f:
        line = f.read().strip()
        # lexical-core with trim_floats(true) might produce "-0"
        assert b"-0" in line


def test_write_negative_zero_validation_pl(tmp_path):
    path = str(tmp_path / "fail_neg_zero.fwf")
    df = pl.DataFrame({"f": [math.copysign(0.0, -1.0)]})

    # Width 1 is NOT enough for "-0"
    specs = [fw.FieldSpec("f", 0, 1, "float")]

    # Manual validation should catch it
    violations = fw.validate_specs_pl(df, specs)
    assert len(violations) > 0
    assert "Column 'f' has data longer" in violations[0]
