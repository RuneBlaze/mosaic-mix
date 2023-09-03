"""
index.py

This script serves as the main entry point for preparing and iterating through a mixed Mosaic dataset using
DataframeMixedMosaicDataModule from the data module. Specifically, it performs the following tasks:

1. Initializes the DataframeMixedMosaicDataModule with specified dataframe and Mosaic shard paths.
2. Calls `prepare_data()` to prepare the underlying data, dumping it into a local DuckDB database.
3. Calls `setup()` with the mode set to "fit" for training.
4. Prints and iterates through the training data loader, displaying each batch for demonstration purposes.

This script is intended for testing and demonstration of the data pipeline and can be adapted for more complex
workflows or integrated into a larger system.
"""
from data import DataframeMixedMosaicDataModule

if __name__ == "__main__":
    data_module = DataframeMixedMosaicDataModule(
        dataframe_path="df.parquet", mds_path="data"
    )
    data_module.prepare_data()
    data_module.setup("fit")
    print("Train dataloader preparing...")
    train_dataloader = data_module.train_dataloader()
    for i, batch in enumerate(train_dataloader):
        print("Batch ", i, ": ", batch)
