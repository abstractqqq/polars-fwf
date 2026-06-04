from __future__ import annotations

import importlib.metadata
import warnings
from typing import TYPE_CHECKING, Sequence

import pyarrow as pa

from ._fwf import DType, ErrorStrategy, FwfParser, FwfReader, FwfWriter, PyFieldSpec

try:
    __version__ = importlib.metadata.version("ffwf")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

if TYPE_CHECKING:
    from ._fwf import PyFieldSpec as FieldSpec

__all__ = [
    "__version__",
    "FwfParser",
    "FieldSpec",
    "DType",
    "ErrorStrategy",
    "read_fwf_arrow",
    "ArrowCapsule",
    "validate_specs_arrow",
    "write_fwf_arrow",
    "infer_specs_arrow",
    "read_fwf_pd",
]


def FieldSpec(
    name: str,
    offset: int,
    length: int,
    dtype: DType | str,
    padding: int | None = None,
    error_strategy: ErrorStrategy = ErrorStrategy.PushNull(),
) -> PyFieldSpec:
    """
    Define a field specification for a fixed-width file.

    Parameters
    ----------
    name : str
        The name of the column.
    offset : int
        The starting byte offset of the field.
    length : int
        The length of the field in bytes.
        For strings, this is the **byte length**, not the character count.
    dtype : DType | str
        The data type of the field. Can be a DType enum or a string alias
        like 'str', 'int', 'f64', etc.
    padding : int | None, optional
        Optional padding byte. Defaults to space.
    error_strategy : ErrorStrategy, optional
        How to handle parsing errors for this field. Defaults to `PushNull()`.
        If using `Fill(bytes)`, the value must be a byte string valid for the
        target dtype and fit within the field length.

    Returns
    -------
    PyFieldSpec
        An internal field specification object.

    Examples
    --------
    >>> import ffwf as fw
    >>> # Basic integer field
    >>> s1 = fw.FieldSpec("id", 0, 5, "int")
    >>> # Integer with Fill strategy (uses 0 as fallback)
    >>> s2 = fw.FieldSpec("val", 5, 5, "int", error_strategy=fw.ErrorStrategy.Fill(b"0    "))
    >>> # String with Fill strategy (uses "N/A" as fallback)
    >>> s3 = fw.FieldSpec("tag", 10, 5, "str", error_strategy=fw.ErrorStrategy.Fill(b"N/A  "))
    """
    if isinstance(dtype, str):
        dtype_lower = dtype.lower()
        if dtype_lower in ("str", "string"):
            resolved_dtype = DType.String
        elif dtype_lower in ("int", "integer", "i32"):
            resolved_dtype = DType.I32
        elif dtype_lower == "i8":
            resolved_dtype = DType.I8
        elif dtype_lower == "i16":
            resolved_dtype = DType.I16
        elif dtype_lower == "i64":
            resolved_dtype = DType.I64
        elif dtype_lower == "u8":
            resolved_dtype = DType.U8
        elif dtype_lower == "u16":
            resolved_dtype = DType.U16
        elif dtype_lower == "u32":
            resolved_dtype = DType.U32
        elif dtype_lower == "u64":
            resolved_dtype = DType.U64
        elif dtype_lower in ("f32", "float"):
            resolved_dtype = DType.F32
        elif dtype_lower in ("f64", "double"):
            resolved_dtype = DType.F64
        else:
            raise ValueError(f"Unknown DType alias: {dtype}")
    else:
        resolved_dtype = dtype

    if length <= 0:
        raise ValueError(f"FieldSpec width must be positive, got {length}")

    # Integer width capacity validation
    max_w = resolved_dtype.max_width()
    if max_w is not None and length > max_w:
        warnings.warn(
            f"Width {length} exceeds maximum capacity for {resolved_dtype} "
            f"(max {max_w} characters). This may cause overflow or parsing errors.",
            UserWarning,
            stacklevel=2,
        )

    return PyFieldSpec(name, offset, length, resolved_dtype, padding, error_strategy)


class ArrowCapsule:
    """
    Internal adapter to bridge Arrow C Data Interface capsules with Arrow/Polars.
    """

    def __init__(self, capsules: tuple):
        """
        Initialize the adapter with a tuple of (schema_capsule, array_capsule).

        Parameters
        ----------
        capsules : tuple
            A tuple containing the Arrow C Data Interface capsules.
        """
        self.schema_capsule, self.array_capsule = capsules

    def __arrow_c_array__(self, requested_schema=None):
        """
        Implement the Arrow C Data Interface protocol.
        """
        return self.schema_capsule, self.array_capsule


def read_fwf_arrow(
    path: str,
    specs: Sequence[PyFieldSpec],
    line_length: int | None = None,
    newline: str | bytes = "\n",
    chunk_size: int | None = None,
    parallel: bool = True,
) -> pa.Table:
    """
    Read a fixed-width file into a PyArrow Table using zero-copy Arrow transfer.

    Parameters
    ----------
    path : str
        Path to the FWF file.
    specs : Sequence[PyFieldSpec]
        List of field specifications defining column names, offsets, lengths, and types.
    line_length : int | None, optional
        The total length of each line in bytes (including newline). If None, it is
        automatically detected.
    newline : str | bytes, default "\\n"
        The newline character(s) used in the file.
    chunk_size : int | None, optional
        The number of rows to parse per batch. If None, it's inferred by the core parser.
    parallel : bool, default True
        Whether to use multi-threaded parsing in the Rust core.

    Returns
    -------
    pa.Table
        A PyArrow Table containing the parsed data.
    """

    # DType to arrow mapping helper
    def _dt_to_pa(dt):
        if dt == DType.I8:
            return pa.int8()
        if dt == DType.I16:
            return pa.int16()
        if dt == DType.I32:
            return pa.int32()
        if dt == DType.I64:
            return pa.int64()
        if dt == DType.U8:
            return pa.uint8()
        if dt == DType.U16:
            return pa.uint16()
        if dt == DType.U32:
            return pa.uint32()
        if dt == DType.U64:
            return pa.uint64()
        if dt == DType.F32:
            return pa.float32()
        if dt == DType.F64:
            return pa.float64()
        if dt == DType.String:
            return pa.utf8()
        return pa.null()

    # Handle empty file
    import os

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        schema = pa.schema([(s.name, _dt_to_pa(s.dtype)) for s in specs])
        return pa.Table.from_batches([], schema=schema)

    newline_bytes = newline if isinstance(newline, bytes) else newline.encode("utf-8")
    stride, data_len = FwfParser.detect_line_length(path, newline_bytes)

    actual_stride = line_length if line_length is not None else stride
    actual_data_len = actual_stride - len(newline_bytes)

    # Validate that all specs are within bounds
    for s in specs:
        if s.offset + s.length > actual_data_len:
            raise ValueError(
                f"FieldSpec '{s.name}' (offset={s.offset}, length={s.length}) "
                f"exceeds data length ({actual_data_len})."
            )

    parser = FwfParser(
        list(specs),
        actual_stride,
        parallel=parallel,
        chunk_size=chunk_size,
    )

    capsule_tuples = parser._parse_path(path)

    if not capsule_tuples:
        schema = pa.schema([(s.name, _dt_to_pa(s.dtype)) for s in specs])
        return pa.Table.from_batches([], schema=schema)

    batches = [pa.record_batch(ArrowCapsule(c)) for c in capsule_tuples]
    return pa.Table.from_batches(batches)


def validate_specs_arrow(table: pa.Table, specs: Sequence[PyFieldSpec]) -> list[str]:
    """
    Validate that the data in the Arrow Table satisfies the provided specifications.

    Checks performed:
    1. **Field Width**: Ensures no value exceeds the specified `length` in bytes.
    2. **Line Breaks**: Ensures no string values contain `\\n` or `\\r`, which
       would break the fixed-width physical layout.

    Parameters
    ----------
    table : pa.Table
        The Arrow Table to validate.
    specs : Sequence[PyFieldSpec]
        The field specifications defining the expected layout.

    Returns
    -------
    list[str]
        A list of violation messages. If empty, all data satisfies the specs.
    """
    import pyarrow.compute as pc

    violations = []
    for s in specs:
        col = table[s.name]
        str_col = pc.cast(col, pa.utf8())

        # 1. Check lengths
        lengths = pc.utf8_length(str_col)
        max_len = pc.max(lengths).as_py() or 0

        if max_len > s.length:
            violations.append(
                f"Column '{s.name}' has data longer ({max_len}) than specified length ({s.length})"
            )

        # 2. Check for newlines/carriages
        if pa.types.is_string(str_col.type):
            n_count = pc.sum(pc.count_substring(str_col, "\n")).as_py() or 0
            r_count = pc.sum(pc.count_substring(str_col, "\r")).as_py() or 0
            if n_count > 0 or r_count > 0:
                violations.append(
                    f"Column '{s.name}' contains {n_count + r_count} line break characters (\\n, \\r) "
                    "which will corrupt the FWF layout."
                )
    return violations


def infer_specs_arrow(
    table: pa.Table,
    decimals: int = 3,
    infer_rows: int = 1000,
) -> list[PyFieldSpec]:
    """
    Automatically infer column widths and types from an Arrow Table.
    """
    import pyarrow.compute as pc

    sample = table.slice(0, infer_rows)
    final_specs = []
    offset = 0

    for name in table.column_names:
        col = sample[name]
        dt = col.type

        # Use DType mapping
        if pa.types.is_integer(dt):
            out_dtype = DType.I64  # Default to largest for inference
        elif pa.types.is_floating(dt):
            out_dtype = DType.F64
        else:
            out_dtype = DType.String

        # Cast to string to find max length
        str_col = pc.cast(col, pa.utf8())
        lengths = pc.utf8_length(str_col)
        max_len = pc.max(lengths).as_py() or 0

        # Heuristic padding
        if pa.types.is_floating(dt):
            length = max_len + 1
        elif pa.types.is_integer(dt):
            length = max_len
        else:
            length = max_len + 5

        final_specs.append(FieldSpec(name, offset, length, out_dtype))
        offset += length
    return final_specs


def write_fwf_arrow(
    table: pa.Table,
    path: str,
    specs: Sequence[PyFieldSpec] | None = None,
    number_padding: str = " ",
    str_padding: str = " ",
    pad_str_end: bool = True,
    decimals: int = 3,
    bool_treatment: tuple[str, str, str] = ("T", "F", "null"),
) -> dict[str, dict]:
    """
    Write a PyArrow Table to a Fixed-Width File (FWF) using a native Rust writer.

    **Note**: This function does **not** perform data validation. If a value exceeds
    the specified field length, it will be silently truncated. If a value contains
    line breaks (\\n, \\r), it will corrupt the FWF layout. To validate your
    data before writing, use :func:`validate_specs_arrow`.

    table : pa.Table
        The PyArrow Table to write.
    path : str
        The path to the output file.
    specs : Sequence[PyFieldSpec] | None, optional
        A sequence of FieldSpec objects defining the output layout.
        If None, the specification is inferred.
    number_padding : str, default " "
        The padding character for numeric columns (right-aligned).
    str_padding : str, default " "
        The padding character for string columns.
    pad_str_end : bool, default True
        If True, string columns are left-aligned (padded at the end).
        If False, string columns are right-aligned (padded at the start).
    decimals : int, default 3
        The precision for float columns. Floats are rounded to this value.
    bool_treatment : tuple[str, str, str], default ("T", "F", "null")
        The string representations for True, False, and Null boolean values.

    Returns
    -------
    dict[str, dict]
        The specification used to write the file.
    """
    if specs is None:
        specs = infer_specs_arrow(table, decimals=decimals)

    num_pad_byte = number_padding.encode("utf-8")[0]
    str_pad_byte = str_padding.encode("utf-8")[0]

    writer = FwfWriter(
        path,
        list(specs),
        num_pad_byte,
        str_pad_byte,
        pad_str_end,
        decimals,
        bool_treatment,
    )

    for batch in table.to_batches():
        # Arrow record batches in Python support the C Data Interface
        # but PyO3 doesn't automatically handle them. We need capsules.
        # However, pa.RecordBatch.__arrow_c_array__ returns the capsules.
        writer.write_batch(batch.__arrow_c_array__())

    writer.flush()

    # Build return spec map
    spec_map = {}
    for s in specs:
        spec_map[s.name] = {
            "offset": s.offset,
            "length": s.length,
            "dtype": str(s.dtype),
        }
    return spec_map


def read_fwf_pd(
    path: str,
    specs: Sequence[PyFieldSpec],
    line_length: int | None = None,
    newline: str | bytes = "\n",
    chunk_size: int | None = None,
    parallel: bool = True,
) -> Any:
    """
    Read a fixed-width file into a Pandas DataFrame.
    """
    table = read_fwf_arrow(
        path,
        specs,
        line_length=line_length,
        newline=newline,
        chunk_size=chunk_size,
        parallel=parallel,
    )
    return table.to_pandas()


# Polars support
try:
    from .polars import (
        read_fwf_pl,
        scan_fwf_pl,
        sink_fwf_pl,
        validate_specs_pl,
        write_fwf_pl,
    )

    __all__ += [
        "read_fwf_pl",
        "scan_fwf_pl",
        "write_fwf_pl",
        "sink_fwf_pl",
        "validate_specs_pl",
    ]
except ImportError:
    pass

# Pandas support
try:
    from .pandas import read_fwf_pd

    __all__ += ["read_fwf_pd"]
except ImportError:
    pass
