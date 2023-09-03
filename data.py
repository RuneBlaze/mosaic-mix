"""
data.py

This module provides utilities and classes for managing mixed Mosaic datasets that combine data from a DuckDB database
and local Mosaic shards. Specifically, it includes:

The DuckdbMixedMosaicDataset class fetches data in a thread-safe manner, and the DataframeMixedMosaicDataModule prepares
the data by dumping it into a local DuckDB database for efficient data retrieval during training.

This module is designed to work with PyTorch Lightning and assumes the underlying data is stored in a specific format
as required by the DuckDB database and Mosaic shards.
"""
from dataclasses import dataclass
from typing import Any, Mapping

import duckdb
import pandas as pd
import pytorch_lightning as pl
from streaming import StreamingDataset
from torch.utils.data import DataLoader, IterableDataset, default_collate

DUMP_DUCKDB_PATH = "data.duckdb"
"""Where the ETL-dumped data is stored locally in the SSD."""


@dataclass
class DictDotAccess(Mapping):
    """Wraps a dictionary. Given dictionary d with keys a, b, c, this class allows access to
    d.a, d.b, d.c via the dot-access syntax."""

    data: dict

    def __post_init__(self):
        # posthoc define_method
        for k, v in self.data.items():
            setattr(self, k, v)

    def __getitem__(self, name):
        return self.data[name]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


def collate_dict_dot_access(batch):
    """Collate function for DictDotAccess objects."""
    return DictDotAccess(default_collate([d.data for d in batch]))


def mark_as_used(variable: Any) -> Any:
    """Utility, to mark a variable as used in the current scope, workaround for linters."""
    return variable


class Cycler:
    """A class that wraps a factory function, and returns an iterator that cycles through the factory function.
    Note that this version does not handle empty iterables."""
    def __init__(self, factory):
        self.factory = factory

    def __iter__(self):
        it = iter(self.factory())
        while True:
            try:
                yield next(it)
            except StopIteration:
                it = iter(self.factory())


class DuckdbMixedMosaicDataset(IterableDataset):
    def __init__(self, db_path: str, mds_path: str):
        super().__init__()
        self.db_path = db_path
        self.mds_path = mds_path
        self.dataset = StreamingDataset(local=self.mds_path)

    def __iter__(self):
        conn = duckdb.connect(
            self.db_path, read_only=True
        )  # readonly allows multiple processes to read the same db
        dataset = self.dataset
        cycler = Cycler(lambda: dataset)
        for mds_record in cycler:
            local_conn = conn.cursor()  # each thread must start its own cursor
            # see https://duckdb.org/docs/guides/python/multiple_threads.html#reader-and-writer-functions for thread safety

            pk = mds_record["id"]
            duckdb_row = (
                local_conn.execute("SELECT * FROM dataset WHERE id = ?", [pk])
                .fetch_df_chunk()
                .to_dict(orient="records")[0]
            )
            mds_record.update(duckdb_row)
            yield DictDotAccess(mds_record)
        conn.close()


class DataframeMixedMosaicDataModule(pl.LightningDataModule):
    """A mixed Mosaic dataset, where the metadata is stored in a dataframe, and the other data is stored in Mosaic shards."""
    def __init__(self, dataframe_path: str, mds_path: str):
        super().__init__()
        self.df_path = dataframe_path
        self.mds_path = mds_path

    def prepare_data(self):
        df = pd.read_parquet(self.df_path)
        mark_as_used(df)
        # dump as duckdb
        conn = duckdb.connect(DUMP_DUCKDB_PATH, read_only=False)
        conn.execute("CREATE TABLE IF NOT EXISTS dataset AS SELECT * FROM df")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS dataset_index ON dataset (id)"
        )  # create index for faster lookup
        conn.close()

    def train_dataloader(self):
        return DataLoader(
            DuckdbMixedMosaicDataset(DUMP_DUCKDB_PATH, self.mds_path),
            batch_size=8,
            num_workers=2,
            collate_fn=collate_dict_dot_access,
        )
