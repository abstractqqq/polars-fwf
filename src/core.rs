use arrow_array::builder::{GenericByteViewBuilder, PrimitiveBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type, StringViewType, UInt8Type,
    UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Arc;

pub static WHITESPACE_LUT: [u8; 256] = {
    let mut table = [0u8; 256];
    table[b' ' as usize] = 1;
    table[b'\t' as usize] = 1;
    table[b'\n' as usize] = 1;
    table[b'\r' as usize] = 1;
    table
};

#[derive(Clone, Copy)]
pub enum DType {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    String,
}

impl DType {
    pub fn to_arrow(&self) -> DataType {
        match self {
            DType::I8 => DataType::Int8,
            DType::I16 => DataType::Int16,
            DType::I32 => DataType::Int32,
            DType::I64 => DataType::Int64,
            DType::U8 => DataType::UInt8,
            DType::U16 => DataType::UInt16,
            DType::U32 => DataType::UInt32,
            DType::U64 => DataType::UInt64,
            DType::F32 => DataType::Float32,
            DType::F64 => DataType::Float64,
            DType::String => DataType::Utf8View,
        }
    }

    pub fn max_width(&self) -> Option<usize> {
        match self {
            DType::I8 => Some(4),
            DType::U8 => Some(3),
            DType::I16 => Some(6),
            DType::U16 => Some(5),
            DType::I32 => Some(11),
            DType::U32 => Some(10),
            DType::I64 => Some(20),
            DType::U64 => Some(20),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum FillValue {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
    String(String),
}

#[derive(Clone, PartialEq)]
#[allow(dead_code)]
pub enum ErrorStrategy {
    PushNull,
    Fill(FillValue),
}

#[derive(Clone)]
pub struct FieldSpec {
    pub name: String,
    pub offset: usize,
    pub length: usize,
    pub dtype: DType,
    pub padding: Option<u8>,
    pub error_strategy: ErrorStrategy,
}

#[derive(Clone, Copy)]
pub enum Par {
    Seq,
    Fixed(usize),
}

pub struct FwfParser {
    pub specs: Vec<FieldSpec>,
    pub line_length: usize,
    pub schema: Arc<Schema>,
    pub chunk_size: usize,
    pub parallelism: Par,
}

impl FwfParser {
    pub fn new(specs: Vec<FieldSpec>, line_length: usize) -> Self {
        let fields: Vec<Field> = specs
            .iter()
            .map(|s| Field::new(&s.name, s.dtype.to_arrow(), true))
            .collect();
        let schema = Arc::new(Schema::new(fields));

        let chunk_size = Self::infer_chunk_size(&specs);

        Self {
            specs,
            line_length,
            schema,
            chunk_size,
            parallelism: Par::Fixed(0),
        }
    }

    pub fn infer_chunk_size(specs: &[FieldSpec]) -> usize {
        let mut est_bytes_per_row = 0;
        for s in specs {
            match s.dtype {
                DType::String => est_bytes_per_row += 16 + s.length,
                _ => est_bytes_per_row += 8,
            }
        }

        let target_batch_size_bytes = 32 * 1024 * 1024; // 32MB target
        let inferred = target_batch_size_bytes / est_bytes_per_row.max(1);
        inferred.clamp(1024, 65536)
    }

    pub fn detect_line_length(path: &str, newline: &[u8]) -> std::io::Result<(usize, usize)> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Use a simple window search for the newline symbol
        let pos = mmap
            .windows(newline.len())
            .position(|window| window == newline)
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Could not detect newline symbol in FWF file",
                )
            })?;

        let stride = pos + newline.len();
        let data_len = pos;
        Ok((stride, data_len))
    }

    pub fn parse_path(&self, path: &str) -> std::io::Result<Vec<RecordBatch>> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(self.parse(&mmap))
    }

    pub fn parse(&self, data: &[u8]) -> Vec<RecordBatch> {
        let total_rows = data.len() / self.line_length;
        if total_rows == 0 {
            return vec![];
        }

        let chunk_size_bytes = self.chunk_size * self.line_length;

        match self.parallelism {
            Par::Seq => data
                .chunks(chunk_size_bytes)
                .map(|chunk| self.parse_batch(chunk))
                .collect(),
            Par::Fixed(n) => {
                let num_threads = if n == 0 {
                    rayon::current_num_threads()
                } else {
                    n.min(rayon::current_num_threads())
                };

                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .build()
                    .unwrap();

                pool.install(|| {
                    data.par_chunks(chunk_size_bytes)
                        .map(|chunk| self.parse_batch(chunk))
                        .collect()
                })
            }
        }
    }

    pub fn parse_batch(&self, data: &[u8]) -> RecordBatch {
        let num_rows = data.len() / self.line_length;
        let mut builders = self
            .specs
            .iter()
            .map(|s| match s.dtype {
                DType::I8 => ColumnBuilder::I8(PrimitiveBuilder::with_capacity(num_rows)),
                DType::I16 => ColumnBuilder::I16(PrimitiveBuilder::with_capacity(num_rows)),
                DType::I32 => ColumnBuilder::I32(PrimitiveBuilder::with_capacity(num_rows)),
                DType::I64 => ColumnBuilder::I64(PrimitiveBuilder::with_capacity(num_rows)),
                DType::U8 => ColumnBuilder::U8(PrimitiveBuilder::with_capacity(num_rows)),
                DType::U16 => ColumnBuilder::U16(PrimitiveBuilder::with_capacity(num_rows)),
                DType::U32 => ColumnBuilder::U32(PrimitiveBuilder::with_capacity(num_rows)),
                DType::U64 => ColumnBuilder::U64(PrimitiveBuilder::with_capacity(num_rows)),
                DType::F32 => ColumnBuilder::F32(PrimitiveBuilder::with_capacity(num_rows)),
                DType::F64 => ColumnBuilder::F64(PrimitiveBuilder::with_capacity(num_rows)),
                DType::String => {
                    ColumnBuilder::String(GenericByteViewBuilder::with_capacity(num_rows))
                }
            })
            .collect::<Vec<_>>();

        for row_idx in 0..num_rows {
            let row_start = row_idx * self.line_length;
            for (spec, builder) in self.specs.iter().zip(builders.iter_mut()) {
                let start = row_start + spec.offset;
                let end = start + spec.length;
                let field = trim_custom(&data[start..end], spec.padding);

                match (builder, &spec.error_strategy) {
                    (ColumnBuilder::I8(b), ErrorStrategy::Fill(FillValue::I8(fill))) => {
                        match lexical_core::parse::<i8>(field) {
                            Ok(v) => b.append_value(v),
                            Err(_) => b.append_value(*fill),
                        }
                    }
                    (ColumnBuilder::I8(b), _) => match lexical_core::parse::<i8>(field) {
                        Ok(v) => b.append_value(v),
                        Err(_) => b.append_null(),
                    },
                    (ColumnBuilder::I16(b), ErrorStrategy::Fill(FillValue::I16(fill))) => {
                        match lexical_core::parse::<i16>(field) {
                            Ok(v) => b.append_value(v),
                            Err(_) => b.append_value(*fill),
                        }
                    }
                    (ColumnBuilder::I16(b), _) => match lexical_core::parse::<i16>(field) {
                        Ok(v) => b.append_value(v),
                        Err(_) => b.append_null(),
                    },
                    (ColumnBuilder::I32(b), ErrorStrategy::Fill(FillValue::I32(fill))) => {
                        match lexical_core::parse::<i32>(field) {
                            Ok(v) => b.append_value(v),
                            Err(_) => b.append_value(*fill),
                        }
                    }
                    (ColumnBuilder::I32(b), _) => match lexical_core::parse::<i32>(field) {
                        Ok(v) => b.append_value(v),
                        Err(_) => b.append_null(),
                    },
                    (ColumnBuilder::I64(b), ErrorStrategy::Fill(FillValue::I64(fill))) => {
                        match lexical_core::parse::<i64>(field) {
                            Ok(v) => b.append_value(v),
                            Err(_) => b.append_value(*fill),
                        }
                    }
                    (ColumnBuilder::I64(b), _) => match lexical_core::parse::<i64>(field) {
                        Ok(v) => b.append_value(v),
                        Err(_) => b.append_null(),
                    },
                    (ColumnBuilder::U8(b), ErrorStrategy::Fill(FillValue::U8(fill))) => {
                        match lexical_core::parse::<u8>(field) {
                            Ok(v) => b.append_value(v),
                            Err(_) => b.append_value(*fill),
                        }
                    }
                    (ColumnBuilder::U8(b), _) => match lexical_core::parse::<u8>(field) {
                        Ok(v) => b.append_value(v),
                        Err(_) => b.append_null(),
                    },
                    (ColumnBuilder::U16(b), ErrorStrategy::Fill(FillValue::U16(fill))) => {
                        match lexical_core::parse::<u16>(field) {
                            Ok(v) => b.append_value(v),
                            Err(_) => b.append_value(*fill),
                        }
                    }
                    (ColumnBuilder::U16(b), _) => match lexical_core::parse::<u16>(field) {
                        Ok(v) => b.append_value(v),
                        Err(_) => b.append_null(),
                    },
                    (ColumnBuilder::U32(b), ErrorStrategy::Fill(FillValue::U32(fill))) => {
                        match lexical_core::parse::<u32>(field) {
                            Ok(v) => b.append_value(v),
                            Err(_) => b.append_value(*fill),
                        }
                    }
                    (ColumnBuilder::U32(b), _) => match lexical_core::parse::<u32>(field) {
                        Ok(v) => b.append_value(v),
                        Err(_) => b.append_null(),
                    },
                    (ColumnBuilder::U64(b), ErrorStrategy::Fill(FillValue::U64(fill))) => {
                        match lexical_core::parse::<u64>(field) {
                            Ok(v) => b.append_value(v),
                            Err(_) => b.append_value(*fill),
                        }
                    }
                    (ColumnBuilder::U64(b), _) => match lexical_core::parse::<u64>(field) {
                        Ok(v) => b.append_value(v),
                        Err(_) => b.append_null(),
                    },
                    (ColumnBuilder::F32(b), ErrorStrategy::Fill(FillValue::F32(fill))) => {
                        match lexical_core::parse::<f32>(field) {
                            Ok(v) => b.append_value(v),
                            Err(_) => b.append_value(*fill),
                        }
                    }
                    (ColumnBuilder::F32(b), _) => match lexical_core::parse::<f32>(field) {
                        Ok(v) => b.append_value(v),
                        Err(_) => b.append_null(),
                    },
                    (ColumnBuilder::F64(b), ErrorStrategy::Fill(FillValue::F64(fill))) => {
                        match lexical_core::parse::<f64>(field) {
                            Ok(v) => b.append_value(v),
                            Err(_) => b.append_value(*fill),
                        }
                    }
                    (ColumnBuilder::F64(b), _) => match lexical_core::parse::<f64>(field) {
                        Ok(v) => b.append_value(v),
                        Err(_) => b.append_null(),
                    },
                    (ColumnBuilder::String(b), ErrorStrategy::Fill(FillValue::String(fill))) => {
                        match std::str::from_utf8(field) {
                            Ok(v) => b.append_value(v),
                            Err(_) => b.append_value(fill),
                        }
                    }
                    (ColumnBuilder::String(b), _) => match std::str::from_utf8(field) {
                        Ok(v) => b.append_value(v),
                        Err(_) => b.append_null(),
                    },
                }
            }
        }

        let arrays = builders
            .into_iter()
            .map(|b| match b {
                ColumnBuilder::I8(mut b) => Arc::new(b.finish()) as ArrayRef,
                ColumnBuilder::I16(mut b) => Arc::new(b.finish()) as ArrayRef,
                ColumnBuilder::I32(mut b) => Arc::new(b.finish()) as ArrayRef,
                ColumnBuilder::I64(mut b) => Arc::new(b.finish()) as ArrayRef,
                ColumnBuilder::U8(mut b) => Arc::new(b.finish()) as ArrayRef,
                ColumnBuilder::U16(mut b) => Arc::new(b.finish()) as ArrayRef,
                ColumnBuilder::U32(mut b) => Arc::new(b.finish()) as ArrayRef,
                ColumnBuilder::U64(mut b) => Arc::new(b.finish()) as ArrayRef,
                ColumnBuilder::F32(mut b) => Arc::new(b.finish()) as ArrayRef,
                ColumnBuilder::F64(mut b) => Arc::new(b.finish()) as ArrayRef,
                ColumnBuilder::String(mut b) => Arc::new(b.finish()) as ArrayRef,
            })
            .collect();

        RecordBatch::try_new(self.schema.clone(), arrays).expect("Failed to create RecordBatch")
    }
}

pub enum ColumnBuilder {
    I8(PrimitiveBuilder<Int8Type>),
    I16(PrimitiveBuilder<Int16Type>),
    I32(PrimitiveBuilder<Int32Type>),
    I64(PrimitiveBuilder<Int64Type>),
    U8(PrimitiveBuilder<UInt8Type>),
    U16(PrimitiveBuilder<UInt16Type>),
    U32(PrimitiveBuilder<UInt32Type>),
    U64(PrimitiveBuilder<UInt64Type>),
    F32(PrimitiveBuilder<Float32Type>),
    F64(PrimitiveBuilder<Float64Type>),
    String(GenericByteViewBuilder<StringViewType>),
}

#[inline(always)]
pub fn trim_ascii_spaces(slice: &[u8]) -> &[u8] {
    let mut start = 0;
    while start < slice.len() && WHITESPACE_LUT[slice[start] as usize] == 1 {
        start += 1;
    }
    let mut end = slice.len();
    while end > start && WHITESPACE_LUT[slice[end - 1] as usize] == 1 {
        end -= 1;
    }
    &slice[start..end]
}

#[inline(always)]
pub fn trim_custom(slice: &[u8], padding: Option<u8>) -> &[u8] {
    match padding {
        None | Some(b' ') => trim_ascii_spaces(slice),
        Some(p) => {
            let mut start = 0;
            while start < slice.len() && slice[start] == p {
                start += 1;
            }
            let mut end = slice.len();
            while end > start && slice[end - 1] == p {
                end -= 1;
            }
            &slice[start..end]
        }
    }
}

pub struct FwfReader {
    mmap: Mmap,
    parser: FwfParser,
    offset: usize,
    burst_size: usize,
}

impl FwfReader {
    pub fn new(
        path: &str,
        specs: Vec<FieldSpec>,
        line_length: usize,
        parallel: Option<bool>,
        chunk_size: Option<usize>,
    ) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut parser = FwfParser::new(specs, line_length);

        let mut burst = 1;
        if let Some(p) = parallel {
            parser.parallelism = if p {
                burst = rayon::current_num_threads().max(1);
                Par::Fixed(0)
            } else {
                Par::Seq
            };
        }

        if let Some(c) = chunk_size {
            parser.chunk_size = c;
        }

        Ok(Self {
            mmap,
            parser,
            offset: 0,
            burst_size: burst,
        })
    }

    pub fn next_burst(&mut self) -> Vec<RecordBatch> {
        if self.offset >= self.mmap.len() {
            return vec![];
        }

        let batch_bytes = self.parser.chunk_size * self.parser.line_length;
        let burst_bytes = batch_bytes * self.burst_size;

        let end = (self.offset + burst_bytes).min(self.mmap.len());

        let actual_end = if end == self.mmap.len() {
            end
        } else {
            self.offset + ((end - self.offset) / self.parser.line_length) * self.parser.line_length
        };

        if actual_end <= self.offset {
            return vec![];
        }

        let batches = self.parser.parse(&self.mmap[self.offset..actual_end]);
        self.offset = actual_end;
        batches
    }
}

pub struct FwfWriter<W: Write> {
    writer: BufWriter<W>,
    specs: Vec<FieldSpec>,
    number_padding: u8,
    str_padding: u8,
    pad_str_end: bool,
    decimals: usize,
    bool_treatment: (String, String, String), // True, False, Null
}

impl<W: Write> FwfWriter<W> {
    pub fn new(
        writer: W,
        specs: Vec<FieldSpec>,
        number_padding: u8,
        str_padding: u8,
        pad_str_end: bool,
        decimals: usize,
        bool_treatment: (String, String, String),
    ) -> Self {
        Self {
            writer: BufWriter::new(writer),
            specs,
            number_padding,
            str_padding,
            pad_str_end,
            decimals,
            bool_treatment,
        }
    }

    pub fn write_batch(&mut self, batch: &RecordBatch) -> std::io::Result<()> {
        let num_rows = batch.num_rows();
        let columns: Vec<(&FieldSpec, &ArrayRef)> = self
            .specs
            .iter()
            .map(|s| {
                (
                    s,
                    batch
                        .column_by_name(&s.name)
                        .unwrap_or_else(|| panic!("Column {} not found in batch", s.name)),
                )
            })
            .collect();

        let mut num_buf = [0u8; 128];
        let float_options = lexical_core::WriteFloatOptions::builder()
            .max_significant_digits(std::num::NonZeroUsize::new(self.decimals))
            .trim_floats(true)
            .build()
            .unwrap();

        for row_idx in 0..num_rows {
            for (spec, col) in &columns {
                let formatted: &[u8] = if col.is_null(row_idx) {
                    &[]
                } else {
                    match col.data_type() {
                        DataType::Int8 => lexical_core::write(
                            col.as_primitive::<Int8Type>().value(row_idx),
                            &mut num_buf,
                        ),
                        DataType::Int16 => lexical_core::write(
                            col.as_primitive::<Int16Type>().value(row_idx),
                            &mut num_buf,
                        ),
                        DataType::Int32 => lexical_core::write(
                            col.as_primitive::<Int32Type>().value(row_idx),
                            &mut num_buf,
                        ),
                        DataType::Int64 => lexical_core::write(
                            col.as_primitive::<Int64Type>().value(row_idx),
                            &mut num_buf,
                        ),
                        DataType::UInt8 => lexical_core::write(
                            col.as_primitive::<UInt8Type>().value(row_idx),
                            &mut num_buf,
                        ),
                        DataType::UInt16 => lexical_core::write(
                            col.as_primitive::<UInt16Type>().value(row_idx),
                            &mut num_buf,
                        ),
                        DataType::UInt32 => lexical_core::write(
                            col.as_primitive::<UInt32Type>().value(row_idx),
                            &mut num_buf,
                        ),
                        DataType::UInt64 => lexical_core::write(
                            col.as_primitive::<UInt64Type>().value(row_idx),
                            &mut num_buf,
                        ),
                        DataType::Float32 => lexical_core::write_with_options::<
                            f32,
                            { lexical_core::format::STANDARD },
                        >(
                            col.as_primitive::<Float32Type>().value(row_idx),
                            &mut num_buf,
                            &float_options,
                        ),
                        DataType::Float64 => lexical_core::write_with_options::<
                            f64,
                            { lexical_core::format::STANDARD },
                        >(
                            col.as_primitive::<Float64Type>().value(row_idx),
                            &mut num_buf,
                            &float_options,
                        ),
                        DataType::Utf8 => col.as_string::<i32>().value(row_idx).as_bytes(),
                        DataType::LargeUtf8 => col.as_string::<i64>().value(row_idx).as_bytes(),
                        DataType::Utf8View => col.as_string_view().value(row_idx).as_bytes(),
                        DataType::Boolean => {
                            if col.as_boolean().value(row_idx) {
                                self.bool_treatment.0.as_bytes()
                            } else {
                                self.bool_treatment.1.as_bytes()
                            }
                        }
                        _ => &[],
                    }
                };

                let formatted = if formatted.is_empty() && col.is_null(row_idx) {
                    self.bool_treatment.2.as_bytes()
                } else {
                    formatted
                };

                let len = formatted.len().min(spec.length);
                let truncated = &formatted[..len];

                let is_numeric = col.data_type().is_numeric();
                let pad_char = if is_numeric {
                    self.number_padding
                } else {
                    self.str_padding
                };

                if is_numeric || !self.pad_str_end {
                    // Right align
                    for _ in 0..(spec.length - len) {
                        self.writer.write_all(&[pad_char])?;
                    }
                    self.writer.write_all(truncated)?;
                } else {
                    // Left align
                    self.writer.write_all(truncated)?;
                    for _ in 0..(spec.length - len) {
                        self.writer.write_all(&[pad_char])?;
                    }
                }
            }
            self.writer.write_all(b"\n")?;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}
