use arrow::datatypes::SchemaRef;
use pyo3::Bound;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyCapsule};
use pyo3_arrow::PyTable;

/// Converts a PyArrow table to an Arrow table using the Arrow C Data Interface
pub fn convert_py_to_arrow_table(py_arrow_table: &Bound<'_, PyAny>) -> PyResult<PyTable> {
    println!("Analyzing PyArrow table...");

    // Try to convert the PyArrow table using the Arrow C Data Interface
    if let Ok(c_stream) = py_arrow_table.getattr("__arrow_c_stream__") {
        // If the PyArrow table has the Arrow C Data Interface, call it to get a capsule
        println!("Found __arrow_c_stream__ method, using C Data Interface...");

        // Call the method and store the result in a variable
        let result = c_stream.call0()?;
        // Downcast to PyCapsule and keep the reference
        let capsule = result.downcast::<PyCapsule>()?;

        // Use from_arrow_pycapsule to create a PyTable
        match PyTable::from_arrow_pycapsule(&capsule) {
            Ok(table) => {
                println!("Successfully converted using Arrow C Data Interface");
                Ok(table)
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "Failed to convert with from_arrow_pycapsule: {:?}",
                e
            ))),
        }
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "PyArrow table doesn't support the Arrow C Data Interface (__arrow_c_stream__)",
        ))
    }
}

/// Prints the schema of an Arrow table
pub fn print_schema(schema: &SchemaRef) {
    println!("\nArrow Table Schema:");
    for field in schema.fields() {
        let type_str = field.data_type().to_string();
        let data_type_str = match type_str.as_str() {
            "Int64" => "int64",
            "Float64" => "double",
            "LargeUtf8" => "large_string",
            dt => dt, // Keep as is for other types
        };
        println!("- {}: {}", field.name(), data_type_str);
    }
}
