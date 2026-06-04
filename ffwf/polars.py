from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Sequence

# Some functionalities work without requiring polars >= 1.34.
import polars as pl
from polars.io.plugins import register_io_source

from . import ArrowCapsule, DType, FwfParser, FwfReader, PyFieldSpec

if TYPE_CHECKING:
    from . import FieldSpec

__all__ = [
    "read_fwf_pl",
    "scan_fwf_pl",
    "write_fwf_pl",
    "sink_fwf_pl",
    "validate_specs_pl",
]

# ==============================================================================
# Validation
# ==============================================================================


def validate_specs_pl(
    df: pl.DataFrame | pl.LazyFrame, specs: Sequence[PyFieldSpec]
) -> list[str]:
    """
    Validate that the data in the Polars DataFrame/LazyFrame satisfies the specs.

    Checks performed:
    1. **Field Width**: Ensures no value exceeds the specified `length` in bytes.
    2. **Line Breaks**: Ensures no string values contain `\\n` or `\\r`, which
       would break the fixed-width physical layout.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        The data to validate.
    specs : Sequence[PyFieldSpec]
        The field specifications.

    Returns
    -------
    list[str]
        A list of violation messages.
    """
    lf = df.lazy()
    # We use Polars expressions to calculate max length and newline presence in parallel
    agg_exprs = []
    for s in specs:
        sc = pl.col(s.name).cast(pl.String)
        agg_exprs.append(sc.str.len_bytes().max().alias(f"{s.name}_len"))
        agg_exprs.append(sc.str.contains(r"[\n\r]").sum().alias(f"{s.name}_newlines"))

    stats = lf.select(agg_exprs).collect()

    violations = []
    for s in specs:
        max_len = stats[f"{s.name}_len"][0] or 0
        newline_count = stats[f"{s.name}_newlines"][0] or 0

        if max_len > s.length:
            violations.append(
                f"Column '{s.name}' has data longer ({max_len}) than specified length ({s.length})"
            )
        if newline_count > 0:
            violations.append(
                f"Column '{s.name}' contains {newline_count} rows with line breaks (\\n, \\r) "
                "which will corrupt the FWF layout."
            )
    return violations


# ==============================================================================
# Polars IO Source (Lazy)
# ==============================================================================


class FwfSource:
    """
    Polars IO Source implementation for Fixed-Width Files (FWF).
    """

    def __init__(
        self,
        path: str,
        specs: Sequence[PyFieldSpec],
        line_length: int,
        chunk_size: int | None,
        parallel: bool = True,
    ):
        """
        Initialize the FWF source.
        """
        self.path = path
        self.specs = specs
        self.line_length = line_length
        self.chunk_size = chunk_size
        self.parallel = parallel

    def __call__(
        self,
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        """
        Execute the IO source and yield DataFrames.
        """
        reader = FwfReader(
            self.path,
            list(self.specs),
            self.line_length,
            parallel=self.parallel,
            chunk_size=batch_size or self.chunk_size,
        )

        count = 0
        while True:
            capsule_tuples = reader.next_burst()
            if not capsule_tuples:
                break

            for capsules in capsule_tuples:
                df = pl.from_arrow(ArrowCapsule(capsules))

                if with_columns:
                    df = df.select(with_columns)

                if predicate is not None:
                    df = df.filter(predicate)

                if n_rows is not None:
                    remaining = n_rows - count
                    if remaining <= 0:
                        return
                    if len(df) > remaining:
                        df = df.head(remaining)

                if len(df) > 0:
                    count += len(df)
                    yield df

                if n_rows is not None and count >= n_rows:
                    return


# ==============================================================================
# Reading (Eager & Lazy)
# ==============================================================================


def _dtype_to_pl(dtype: DType) -> pl.DataType:
    """
    Convert an internal DType to a Polars DataType.
    """
    if dtype == DType.I8:
        return pl.Int8
    if dtype == DType.I16:
        return pl.Int16
    if dtype == DType.I32:
        return pl.Int32
    if dtype == DType.I64:
        return pl.Int64
    if dtype == DType.U8:
        return pl.UInt8
    if dtype == DType.U16:
        return pl.UInt16
    if dtype == DType.U32:
        return pl.UInt32
    if dtype == DType.U64:
        return pl.UInt64
    if dtype == DType.F32:
        return pl.Float32
    if dtype == DType.F64:
        return pl.Float64
    if dtype == DType.String:
        return pl.String
    raise ValueError(f"Unknown DType: {dtype}")


def read_fwf_pl(
    path: str,
    specs: Sequence[PyFieldSpec],
    line_length: int | None = None,
    newline: str | bytes = "\n",
    chunk_size: int | None = None,
    parallel: bool = True,
) -> pl.DataFrame:
    """
    Read a fixed-width file into a Polars DataFrame.
    """
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
        # Return empty frame with correct schema
        return pl.DataFrame(schema={s.name: _dtype_to_pl(s.dtype) for s in specs})

    dfs = [pl.from_arrow(ArrowCapsule(c)) for c in capsule_tuples]
    return pl.concat(dfs, how="vertical")


def scan_fwf_pl(
    path: str,
    specs: Sequence[PyFieldSpec],
    line_length: int | None = None,
    newline: str | bytes = "\n",
    chunk_size: int | None = None,
    parallel: bool = True,
) -> pl.LazyFrame:
    """
    Lazily scan a fixed-width file into a Polars LazyFrame.
    """
    newline_bytes = newline if isinstance(newline, bytes) else newline.encode("utf-8")

    if line_length is None:
        line_length, _ = FwfParser.detect_line_length(path, newline_bytes)

    schema = pl.Schema({s.name: _dtype_to_pl(s.dtype) for s in specs})

    return register_io_source(
        io_source=FwfSource(path, specs, line_length, chunk_size, parallel),
        schema=schema,
    )


# ==============================================================================
# Writing (Eager & Streaming)
# ==============================================================================


def write_fwf_pl(
    df: pl.DataFrame | pl.LazyFrame,
    path: str,
    specs: Sequence[PyFieldSpec] | None = None,
    number_padding: str = " ",
    str_padding: str = " ",
    pad_str_end: bool = True,
    decimals: int = 3,
    bool_treatment: tuple[str, str, str] = ("T", "F", "null"),
    simple_dtypes: bool = True,
) -> dict[str, dict]:
    """
    Write a Polars DataFrame or LazyFrame to a Fixed-Width File (FWF) eagerly.

    **Note**: This function does **not** perform data validation. If a value exceeds
    the specified field length, it will be silently truncated. If a value contains
    line breaks (\\n, \\r), it will corrupt the FWF layout. To validate your
    data before writing, use :func:`validate_specs_pl`.
    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        The data to write.
    path : str
        The path to the output file.
    specs : Sequence[PyFieldSpec] | None, optional
        The field specifications. If None, widths and types are inferred.
    number_padding : str, default " "
        The padding character for numeric columns (right-aligned).
    str_padding : str, default " "
        The padding character for string columns.
    pad_str_end : bool, default True
        If True, string columns are left-aligned (padded at the end).
    decimals : int, default 3
        The precision for float columns.
    bool_treatment : tuple[str, str, str], default ("T", "F", "null")
        The string representations for True, False, and Null boolean values.
    simple_dtypes : bool, default True
        If True, use simplified dtype names in the returned spec map.
    """
    from . import infer_specs_arrow, write_fwf_arrow

    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    table = df.to_arrow()

    if specs is None:
        specs = infer_specs_arrow(table, decimals=decimals)

    res_specs = write_fwf_arrow(
        table,
        path,
        specs=specs,
        number_padding=number_padding,
        str_padding=str_padding,
        pad_str_end=pad_str_end,
        decimals=decimals,
        bool_treatment=bool_treatment,
    )

    if simple_dtypes:
        for name, info in res_specs.items():
            dt = info["dtype"]
            if dt.startswith("I") or dt.startswith("U") or "int" in dt.lower():
                info["dtype"] = "int"
            elif dt.startswith("F") or "float" in dt.lower():
                info["dtype"] = "f64"
            elif dt == "String" or "str" in dt.lower():
                info["dtype"] = "str"
    return res_specs


def sink_fwf_pl(
    lf: pl.LazyFrame,
    path: str,
    specs: Sequence[PyFieldSpec] | None = None,
    number_padding: str = " ",
    str_padding: str = " ",
    pad_str_end: bool = True,
    decimals: int = 3,
    bool_treatment: tuple[str, str, str] = ("T", "F", "null"),
    simple_dtypes: bool = True,
    infer_specs_rows: int | None = 1000,
) -> dict[str, dict]:
    """
    Stream a Polars LazyFrame to a Fixed-Width File (FWF).

    **Note**: This function does **not** perform data validation. If a value exceeds
    the specified field length, it will be silently truncated. If a value contains
    line breaks (\\n, \\r), it will corrupt the FWF layout. To validate your
    data before writing, use :func:`validate_specs_pl`.
    Parameters
    ----------
    lf : pl.LazyFrame
        The LazyFrame to write.
    path : str
        The path to the output file.
    specs : Sequence[PyFieldSpec] | None, optional
        The field specifications. If None, widths and types are inferred.
    number_padding : str, default " "
        The padding character for numeric columns.
    str_padding : str, default " "
        The padding character for string columns.
    pad_str_end : bool, default True
        Alignment for string columns.
    decimals : int, default 3
        Precision for float columns.
    bool_treatment : tuple[str, str, str], default ("T", "F", "null")
        Representations for booleans and nulls.
    simple_dtypes : bool, default True
        Simplified dtype names in output.
    infer_specs_rows : int | None, default 1000
        Number of rows for schema inference.
    """
    from . import FwfWriter, infer_specs_arrow

    if specs is None:
        sample = lf.head(infer_specs_rows or 1000).collect().to_arrow()
        specs = infer_specs_arrow(sample, decimals=decimals)

    num_pad_byte = number_padding.encode("utf-8")[0]
    str_pad_byte = str_padding.encode("utf-8")[0]

    writer = FwfWriter(
        path,
        list(specs),
        num_pad_byte,
        str_pad_byte,
        pad_str_end,
        decimals,
    )

    for batch in lf.collect_batches():
        # Polars batch is a DataFrame, convert to Arrow RecordBatch capsules
        rb = batch.to_arrow().to_batches()[0]
        writer.write_batch(rb.__arrow_c_array__())

    writer.flush()

    res_specs = {}
    for s in specs:
        dt = str(s.dtype)
        if simple_dtypes:
            if dt.startswith("I") or dt.startswith("U") or "int" in dt.lower():
                dt = "int"
            elif dt.startswith("F") or "float" in dt.lower():
                dt = "f64"
            elif dt == "String" or "str" in dt.lower():
                dt = "str"

        res_specs[s.name] = {
            "offset": s.offset,
            "length": s.length,
            "dtype": dt,
        }
    return res_specs
