# Import necessary libraries
import dask_cudf
import dask.dataframe as dd
import os
import sys

# Define function to remove all-zero columns
def remove_all_zero_columns(dataframe):
    non_zero_columns = [col for col in dataframe.columns if not (dataframe[col] == 0).all()]
    return dataframe[non_zero_columns]

# Set your dask directory path here
dask_directory_path = "path/to/your/dask_directory"

# Read in the dask directory with multiple partitions in CSV format
ddf = dask_cudf.read_csv(f"{sys.argv[1]}/*").set_index("Unnamed: 0")

# Remove all-zero columns using dask_cudf
non_zero_columns_ddf = ddf.map_partitions(remove_all_zero_columns)

# Trigger computation and convert the dask_cudf DataFrame to a cudf DataFrame
non_zero_columns_cudf = non_zero_columns_ddf.compute()

# Save the resulting cudf DataFrame as a new CSV file
non_zero_columns_cudf.to_csv("output/filtered_data.csv", index=False)
