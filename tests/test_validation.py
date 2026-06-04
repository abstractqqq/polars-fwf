import polars as pl
import pytest

import ffwf as fw


def test_fieldspec_width_validation_pl():
    # Negative width
    with pytest.raises(ValueError, match="width must be positive"):
        fw.FieldSpec("a", 0, -1, "int")

    # Zero width
    with pytest.raises(ValueError, match="width must be positive"):
        fw.FieldSpec("a", 0, 0, "int")


def test_fieldspec_integer_capacity_pl():
    # I8 max 4 chars (-128)
    fw.FieldSpec("a", 0, 4, "i8")  # OK
    with pytest.warns(UserWarning, match="exceeds maximum capacity for I8"):
        fw.FieldSpec("a", 0, 5, "i8")

    # U8 max 3 chars (255)
    fw.FieldSpec("a", 0, 3, "u8")  # OK
    with pytest.warns(UserWarning, match="exceeds maximum capacity for U8"):
        fw.FieldSpec("a", 0, 4, "u8")

    # I16 max 6 chars (-32768)
    fw.FieldSpec("a", 0, 6, "i16")  # OK
    with pytest.warns(UserWarning, match="exceeds maximum capacity for I16"):
        fw.FieldSpec("a", 0, 7, "i16")

    # I32 max 11 chars
    fw.FieldSpec("a", 0, 11, "i32")  # OK
    fw.FieldSpec("a", 0, 11, "int")  # OK
    fw.FieldSpec("a", 0, 11, "integer")  # OK
    with pytest.warns(UserWarning, match="exceeds maximum capacity for I32"):
        fw.FieldSpec("a", 0, 12, "i32")
    with pytest.warns(UserWarning, match="exceeds maximum capacity for I32"):
        fw.FieldSpec("a", 0, 12, "int")
    with pytest.warns(UserWarning, match="exceeds maximum capacity for I32"):
        fw.FieldSpec("a", 0, 12, "integer")


def test_write_capacity_warning_pl(tmp_path):
    path = str(tmp_path / "warn.fwf")
    df = pl.DataFrame({"a": [1]})

    # Spec width 12 for I32 should warn
    with pytest.warns(UserWarning, match="exceeds maximum capacity"):
        specs = [fw.FieldSpec("a", 0, 12, "i32")]

    fw.write_fwf_pl(df, path, specs=specs)

    # Inference that results in large width should also warn
    # Let's use a very long string to trigger large width inference
    df_large = pl.DataFrame({"a": ["x" * 50]})
    # Wait, inference for strings doesn't have a max capacity warning.
    # Only integers do.
    # Let's just test FieldSpec directly for the second case.
    with pytest.warns(UserWarning, match="exceeds maximum capacity"):
        fw.FieldSpec("a", 0, 21, "i64")


if __name__ == "__main__":
    pytest.main([__file__])
