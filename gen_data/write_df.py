# Import necessary libraries
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Define the schema for the table
schema = pa.schema(
    [
        pa.field("id", pa.int64()),  # 'id' column with 64-bit integer type
        pa.field("x", pa.float64()),  # 'x' column with 64-bit float type
        pa.field("y", pa.float64()),  # 'y' column with 64-bit float type
        pa.field("marker_string", pa.string()),
    ]
)

# Define the number of rows
N = 20000

# Create a table using the schema and arrays
table = pa.Table.from_arrays(
    [
        pa.array(range(N)),  # Array of numbers from 0 to 9
        pa.array(np.random.randn(N)),  # Random float numbers using numpy
        pa.array(np.random.randn(N)),  # Another set of random float numbers using numpy
        pa.array([str(i) for i in range(N)]),  # Random string from the list
    ],
    schema=schema,  # Using the defined schema
)

# Specify the output path for the Parquet file
parquet_path = "df.parquet"

# Write the table to a Parquet file
pq.write_table(table, parquet_path)
