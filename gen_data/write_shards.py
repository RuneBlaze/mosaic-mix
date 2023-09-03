"""Script to generate shards"""

from streaming import MDSWriter

columns = {
    "id": "int",
    "dummy_text": "str",
}

shards_dir = "data"

with MDSWriter(
    out=shards_dir, columns=columns, compression="zstd", size_limit=1024 * 5
) as writer:
    # artificially set it as 5kb per shard
    for i in range(1, 10000):
        writer.write(
            {
                "id": i,
                "dummy_text": f"Dummy text {i}",
            }
        )
