mod core;

use arrow_array::ffi::{FFI_ArrowArray, FFI_ArrowSchema, from_ffi};
use arrow_array::{Array, RecordBatch, StructArray};
use pyo3::prelude::*;
use std::fs::File;

#[pyclass(name = "DType", eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PyDType {
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

#[pymethods]
impl PyDType {
    fn __str__(&self) -> &str {
        match self {
            PyDType::I8 => "I8",
            PyDType::I16 => "I16",
            PyDType::I32 => "I32",
            PyDType::I64 => "I64",
            PyDType::U8 => "U8",
            PyDType::U16 => "U16",
            PyDType::U32 => "U32",
            PyDType::U64 => "U64",
            PyDType::F32 => "F32",
            PyDType::F64 => "F64",
            PyDType::String => "String",
        }
    }

    fn max_width(&self) -> Option<usize> {
        let core_dtype: core::DType = (*self).into();
        core_dtype.max_width()
    }
}

impl From<PyDType> for core::DType {
    fn from(dtype: PyDType) -> Self {
        match dtype {
            PyDType::I8 => core::DType::I8,
            PyDType::I16 => core::DType::I16,
            PyDType::I32 => core::DType::I32,
            PyDType::I64 => core::DType::I64,
            PyDType::U8 => core::DType::U8,
            PyDType::U16 => core::DType::U16,
            PyDType::U32 => core::DType::U32,
            PyDType::U64 => core::DType::U64,
            PyDType::F32 => core::DType::F32,
            PyDType::F64 => core::DType::F64,
            PyDType::String => core::DType::String,
        }
    }
}

#[pyclass(name = "ErrorStrategy", eq)]
#[derive(Clone, PartialEq)]
pub enum PyErrorStrategy {
    PushNull(),
    Fill(Vec<u8>),
}

impl From<PyErrorStrategy> for core::ErrorStrategy {
    fn from(s: PyErrorStrategy) -> Self {
        match s {
            PyErrorStrategy::PushNull() => core::ErrorStrategy::PushNull,
            PyErrorStrategy::Fill(_) => {
                panic!("PyErrorStrategy::Fill must be converted manually with dtype context")
            }
        }
    }
}

#[pymethods]
impl PyErrorStrategy {
    fn __str__(&self) -> String {
        match self {
            PyErrorStrategy::PushNull() => "PushNull".to_string(),
            PyErrorStrategy::Fill(v) => format!("Fill({:?})", v),
        }
    }
}

#[pyclass(name = "PyFieldSpec")]
#[derive(Clone)]
pub struct PyFieldSpec {
    /// The name of the field.
    #[pyo3(get, set)]
    pub name: String,
    /// The byte offset from the start of the line where this field begins.
    #[pyo3(get, set)]
    pub offset: usize,
    /// The length of the field in bytes.
    ///
    /// For `DType::String`, this is the **byte length**, not the character count.
    #[pyo3(get)]
    pub length: usize,
    /// The data type of the field.
    #[pyo3(get)]
    pub dtype: PyDType,
    /// Optional byte used for padding/trimming. Defaults to space (0x20).
    #[pyo3(get, set)]
    pub padding: Option<u8>,
    /// The strategy used when a parsing error occurs.
    #[pyo3(get)]
    pub error_strategy: PyErrorStrategy,
}

#[pymethods]
impl PyFieldSpec {
    #[new]
    #[pyo3(signature = (name, offset, length, dtype, padding=None, error_strategy=PyErrorStrategy::PushNull()))]
    pub fn new(
        name: String,
        offset: usize,
        length: usize,
        dtype: PyDType,
        padding: Option<u8>,
        error_strategy: PyErrorStrategy,
    ) -> PyResult<Self> {
        validate_fill_strategy(&dtype, &error_strategy, length)?;
        Ok(Self {
            name,
            offset,
            length,
            dtype,
            padding,
            error_strategy,
        })
    }

    #[setter]
    pub fn set_length(&mut self, length: usize) -> PyResult<()> {
        validate_fill_strategy(&self.dtype, &self.error_strategy, length)?;
        self.length = length;
        Ok(())
    }

    #[setter]
    pub fn set_dtype(&mut self, dtype: PyDType) -> PyResult<()> {
        validate_fill_strategy(&dtype, &self.error_strategy, self.length)?;
        self.dtype = dtype;
        Ok(())
    }

    #[setter]
    pub fn set_error_strategy(&mut self, error_strategy: PyErrorStrategy) -> PyResult<()> {
        validate_fill_strategy(&self.dtype, &error_strategy, self.length)?;
        self.error_strategy = error_strategy;
        Ok(())
    }

    #[getter]
    pub fn get_error_strategy(&self) -> PyErrorStrategy {
        self.error_strategy.clone()
    }
}

fn validate_fill_strategy(
    dtype: &PyDType,
    strategy: &PyErrorStrategy,
    length: usize,
) -> PyResult<()> {
    match strategy {
        PyErrorStrategy::PushNull() => Ok(()),
        PyErrorStrategy::Fill(bytes) => {
            if bytes.len() > length {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Fill value length ({}) exceeds field length ({})",
                    bytes.len(),
                    length
                )));
            }
            let core_dtype: core::DType = (*dtype).into();
            let field = core::trim_ascii_spaces(bytes);
            let valid = match core_dtype {
                core::DType::I8 => lexical_core::parse::<i8>(field).is_ok(),
                core::DType::I16 => lexical_core::parse::<i16>(field).is_ok(),
                core::DType::I32 => lexical_core::parse::<i32>(field).is_ok(),
                core::DType::I64 => lexical_core::parse::<i64>(field).is_ok(),
                core::DType::U8 => lexical_core::parse::<u8>(field).is_ok(),
                core::DType::U16 => lexical_core::parse::<u16>(field).is_ok(),
                core::DType::U32 => lexical_core::parse::<u32>(field).is_ok(),
                core::DType::U64 => lexical_core::parse::<u64>(field).is_ok(),
                core::DType::F32 => lexical_core::parse::<f32>(field).is_ok(),
                core::DType::F64 => lexical_core::parse::<f64>(field).is_ok(),
                core::DType::String => std::str::from_utf8(field).is_ok(),
            };
            if !valid {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Cannot parse fill value {:?} as {:?}",
                    bytes, dtype
                )));
            }
            Ok(())
        }
    }
}

fn parse_fill_value(dtype: &PyDType, bytes: &[u8]) -> core::FillValue {
    let core_dtype: core::DType = (*dtype).into();
    let field = core::trim_ascii_spaces(bytes);
    match core_dtype {
        core::DType::I8 => core::FillValue::I8(lexical_core::parse::<i8>(field).unwrap()),
        core::DType::I16 => core::FillValue::I16(lexical_core::parse::<i16>(field).unwrap()),
        core::DType::I32 => core::FillValue::I32(lexical_core::parse::<i32>(field).unwrap()),
        core::DType::I64 => core::FillValue::I64(lexical_core::parse::<i64>(field).unwrap()),
        core::DType::U8 => core::FillValue::U8(lexical_core::parse::<u8>(field).unwrap()),
        core::DType::U16 => core::FillValue::U16(lexical_core::parse::<u16>(field).unwrap()),
        core::DType::U32 => core::FillValue::U32(lexical_core::parse::<u32>(field).unwrap()),
        core::DType::U64 => core::FillValue::U64(lexical_core::parse::<u64>(field).unwrap()),
        core::DType::F32 => core::FillValue::F32(lexical_core::parse::<f32>(field).unwrap()),
        core::DType::F64 => core::FillValue::F64(lexical_core::parse::<f64>(field).unwrap()),
        core::DType::String => {
            core::FillValue::String(std::str::from_utf8(field).unwrap().to_string())
        }
    }
}

#[pyclass(name = "FwfParser")]
pub struct PyFwfParser {
    inner: core::FwfParser,
}

#[pymethods]
impl PyFwfParser {
    #[new]
    #[pyo3(signature = (specs, line_length, parallel=None, chunk_size=None))]
    pub fn new(
        specs: Vec<PyFieldSpec>,
        line_length: usize,
        parallel: Option<bool>,
        chunk_size: Option<usize>,
    ) -> Self {
        let core_specs = specs
            .into_iter()
            .map(|s| {
                let strategy = match s.error_strategy {
                    PyErrorStrategy::PushNull() => core::ErrorStrategy::PushNull,
                    PyErrorStrategy::Fill(ref bytes) => {
                        core::ErrorStrategy::Fill(parse_fill_value(&s.dtype, bytes))
                    }
                };
                core::FieldSpec {
                    name: s.name,
                    offset: s.offset,
                    length: s.length,
                    dtype: s.dtype.into(),
                    padding: s.padding,
                    error_strategy: strategy,
                }
            })
            .collect();
        let mut inner = core::FwfParser::new(core_specs, line_length);
        if let Some(p) = parallel {
            inner.parallelism = if p {
                core::Par::Fixed(0)
            } else {
                core::Par::Seq
            };
        }
        if let Some(c) = chunk_size {
            inner.chunk_size = c;
        }
        Self { inner }
    }

    pub fn set_chunk_size(&mut self, chunk_size: usize) {
        self.inner.chunk_size = chunk_size;
    }

    #[staticmethod]
    pub fn detect_line_length(path: &str, newline: &[u8]) -> PyResult<(usize, usize)> {
        core::FwfParser::detect_line_length(path, newline)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{e}")))
    }

    #[staticmethod]
    pub fn infer_chunk_size(specs: Vec<PyFieldSpec>) -> usize {
        let core_specs: Vec<core::FieldSpec> = specs
            .into_iter()
            .map(|s| {
                let strategy = match s.error_strategy {
                    PyErrorStrategy::PushNull() => core::ErrorStrategy::PushNull,
                    PyErrorStrategy::Fill(ref bytes) => {
                        core::ErrorStrategy::Fill(parse_fill_value(&s.dtype, bytes))
                    }
                };
                core::FieldSpec {
                    name: s.name,
                    offset: s.offset,
                    length: s.length,
                    dtype: s.dtype.into(),
                    padding: s.padding,
                    error_strategy: strategy,
                }
            })
            .collect();
        core::FwfParser::infer_chunk_size(&core_specs)
    }

    pub fn parse(&self, py: Python, data: &[u8]) -> PyResult<Vec<PyObject>> {
        let batches = self.inner.parse(data);
        let mut py_batches = Vec::with_capacity(batches.len());
        for batch in batches {
            py_batches.push(record_batch_to_capsule(py, batch)?);
        }
        Ok(py_batches)
    }

    pub fn _parse_path(&self, py: Python, path: &str) -> PyResult<Vec<PyObject>> {
        let batches = self
            .inner
            .parse_path(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{e}")))?;
        let mut py_batches = Vec::with_capacity(batches.len());
        for batch in batches {
            py_batches.push(record_batch_to_capsule(py, batch)?);
        }
        Ok(py_batches)
    }
}

#[pyclass(name = "FwfReader")]
pub struct PyFwfReader {
    inner: core::FwfReader,
}

#[pymethods]
impl PyFwfReader {
    #[new]
    #[pyo3(signature = (path, specs, line_length, parallel=None, chunk_size=None))]
    pub fn new(
        path: &str,
        specs: Vec<PyFieldSpec>,
        line_length: usize,
        parallel: Option<bool>,
        chunk_size: Option<usize>,
    ) -> PyResult<Self> {
        let core_specs = specs
            .into_iter()
            .map(|s| {
                let strategy = match s.error_strategy {
                    PyErrorStrategy::PushNull() => core::ErrorStrategy::PushNull,
                    PyErrorStrategy::Fill(ref bytes) => {
                        core::ErrorStrategy::Fill(parse_fill_value(&s.dtype, bytes))
                    }
                };
                core::FieldSpec {
                    name: s.name,
                    offset: s.offset,
                    length: s.length,
                    dtype: s.dtype.into(),
                    padding: s.padding,
                    error_strategy: strategy,
                }
            })
            .collect();
        let inner = core::FwfReader::new(path, core_specs, line_length, parallel, chunk_size)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{e}")))?;
        Ok(Self { inner })
    }

    pub fn next_burst(&mut self, py: Python) -> PyResult<Vec<PyObject>> {
        let batches = self.inner.next_burst();
        let mut py_batches = Vec::with_capacity(batches.len());
        for batch in batches {
            py_batches.push(record_batch_to_capsule(py, batch)?);
        }
        Ok(py_batches)
    }
}

#[pyclass(name = "FwfWriter")]
pub struct PyFwfWriter {
    inner: core::FwfWriter<File>,
}

#[pymethods]
impl PyFwfWriter {
    #[new]
    #[pyo3(signature = (path, specs, number_padding=b' ', str_padding=b' ', pad_str_end=true, decimals=3, bool_treatment=("T".to_string(), "F".to_string(), "null".to_string())))]
    pub fn new(
        path: &str,
        specs: Vec<PyFieldSpec>,
        number_padding: u8,
        str_padding: u8,
        pad_str_end: bool,
        decimals: usize,
        bool_treatment: (String, String, String),
    ) -> PyResult<Self> {
        let core_specs = specs
            .into_iter()
            .map(|s| {
                let strategy = match s.error_strategy {
                    PyErrorStrategy::PushNull() => core::ErrorStrategy::PushNull,
                    PyErrorStrategy::Fill(ref bytes) => {
                        core::ErrorStrategy::Fill(parse_fill_value(&s.dtype, bytes))
                    }
                };
                core::FieldSpec {
                    name: s.name,
                    offset: s.offset,
                    length: s.length,
                    dtype: s.dtype.into(),
                    padding: s.padding,
                    error_strategy: strategy,
                }
            })
            .collect();

        let file = File::create(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{e}")))?;

        let inner = core::FwfWriter::new(
            file,
            core_specs,
            number_padding,
            str_padding,
            pad_str_end,
            decimals,
            bool_treatment,
        );
        Ok(Self { inner })
    }

    pub fn write_batch(&mut self, py: Python, capsules: (PyObject, PyObject)) -> PyResult<()> {
        let batch = capsule_to_record_batch(py, capsules)?;
        self.inner
            .write_batch(&batch)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{e}")))
    }

    pub fn flush(&mut self) -> PyResult<()> {
        self.inner
            .flush()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{e}")))
    }
}

unsafe extern "C" fn array_destructor(capsule: *mut pyo3::ffi::PyObject) {
    let ptr = unsafe { pyo3::ffi::PyCapsule_GetPointer(capsule, c"arrow_array".as_ptr()) };
    if !ptr.is_null() {
        let _ = unsafe { Box::from_raw(ptr as *mut FFI_ArrowArray) };
    }
}

unsafe extern "C" fn schema_destructor(capsule: *mut pyo3::ffi::PyObject) {
    let ptr = unsafe { pyo3::ffi::PyCapsule_GetPointer(capsule, c"arrow_schema".as_ptr()) };
    if !ptr.is_null() {
        let _ = unsafe { Box::from_raw(ptr as *mut FFI_ArrowSchema) };
    }
}

fn record_batch_to_capsule(py: Python, batch: RecordBatch) -> PyResult<PyObject> {
    let struct_array: StructArray = batch.into();
    let array_data = struct_array.to_data();

    let ffi_array = FFI_ArrowArray::new(&array_data);
    let ffi_schema = FFI_ArrowSchema::try_from(array_data.data_type())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{e}")))?;

    let array_ptr = Box::into_raw(Box::new(ffi_array));
    let schema_ptr = Box::into_raw(Box::new(ffi_schema));

    unsafe {
        let array_capsule = pyo3::ffi::PyCapsule_New(
            array_ptr as *mut _,
            c"arrow_array".as_ptr(),
            Some(array_destructor),
        );
        let schema_capsule = pyo3::ffi::PyCapsule_New(
            schema_ptr as *mut _,
            c"arrow_schema".as_ptr(),
            Some(schema_destructor),
        );

        if array_capsule.is_null() || schema_capsule.is_null() {
            return Err(PyErr::fetch(py));
        }

        let array_obj = PyObject::from_owned_ptr(py, array_capsule);
        let schema_obj = PyObject::from_owned_ptr(py, schema_capsule);

        Ok((schema_obj, array_obj).into_py(py))
    }
}

fn capsule_to_record_batch(py: Python, capsules: (PyObject, PyObject)) -> PyResult<RecordBatch> {
    let (schema_obj, array_obj) = capsules;

    unsafe {
        let array_ptr =
            pyo3::ffi::PyCapsule_GetPointer(array_obj.as_ptr(), c"arrow_array".as_ptr());
        let schema_ptr =
            pyo3::ffi::PyCapsule_GetPointer(schema_obj.as_ptr(), c"arrow_schema".as_ptr());

        if array_ptr.is_null() || schema_ptr.is_null() {
            return Err(PyErr::fetch(py));
        }

        let ffi_array = std::ptr::read(array_ptr as *const FFI_ArrowArray);
        let ffi_schema = std::ptr::read(schema_ptr as *const FFI_ArrowSchema);

        // Disarm the destructors in the capsules.
        // This is necessary because from_ffi will take ownership and eventually call the release callback.
        // If the capsule also has a destructor that calls release or frees the struct, it would double-free.
        pyo3::ffi::PyCapsule_SetDestructor(array_obj.as_ptr(), None);
        pyo3::ffi::PyCapsule_SetDestructor(schema_obj.as_ptr(), None);

        let array_data = from_ffi(ffi_array, &ffi_schema)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{e}")))?;

        let struct_array = StructArray::from(array_data);
        Ok(RecordBatch::from(&struct_array))
    }
}

#[pymodule]
fn _fwf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDType>()?;
    m.add_class::<PyErrorStrategy>()?;
    m.add_class::<PyFieldSpec>()?;
    m.add_class::<PyFwfParser>()?;
    m.add_class::<PyFwfReader>()?;
    m.add_class::<PyFwfWriter>()?;
    Ok(())
}
