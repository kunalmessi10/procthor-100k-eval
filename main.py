import gzip
import os
from tqdm import tqdm
import prior
import urllib.request

try:
    from prior import LazyJsonDataset
except:
    raise ImportError("Please update the prior package (pip install --upgrade prior).")


def load_dataset() -> prior.DatasetDict:
    """Load the houses dataset."""
    data = {}
    # for split, size in zip(("val", "test"), (100_000, 1_000, 1_000)):

    for split in ("val"):
        if not f"procthor_100k_objectnav_{split}.jsonl.gz" in os.listdir("./"):
            url = f"https://prior-datasets.s3.us-east-2.amazonaws.com/procthor_100k_eval-kunal/procthor_100k_objectnav_val.jsonl.gz"
            urllib.request.urlretrieve(
                url, "./procthor_100k_objectnav_{}.jsonl.gz".format(split)
            )
        with gzip.open(f"procthor_100k_objectnav_{split}.jsonl.gz", "r") as f:
            tasks = [line for line in tqdm(f, desc=f"Loading {split}")]
        data[split] = LazyJsonDataset(
            data=tasks, dataset="procthor-100k-eval", split=split
        )
    return prior.DatasetDict(**data)
