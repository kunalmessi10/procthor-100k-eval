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
    base_url = (
        "https://prior-datasets.s3.us-east-2.amazonaws.com/procthor_100k_v01_benchmark/"
    )

    for split in ["val", "test"]:
        for task in ["objectnav", "fetch2room", "objexplore"]:
            filename = f"procthor_100k_{task}_{split}.jsonl.gz"
            print(filename)
            if not filename in os.listdir("./"):
                split_task_list = []
                try:
                    response = urllib.request.urlopen(base_url)
                    urllib.request.urlretrieve(
                        base_url + filename,
                        "./" + filename,
                    )
                except urllib.error.URLError as e:
                    print(f"Error: {task} for {split} not found")
                    continue

            with gzip.open(filename, "r") as f:
                tasks = [line for line in tqdm(f, desc=f"Loading {split}")]

            split_task_list = split_task_list + tasks

        data[split] = LazyJsonDataset(
            data=split_task_list, dataset="procthor-100k-eval", split=split
        )
    return prior.DatasetDict(**data)
