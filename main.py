import gzip
from tqdm import tqdm
import prior

try:
    from prior import LazyJsonDataset
except:
    raise ImportError("Please update the prior package (pip install --upgrade prior).")


def load_dataset() -> prior.DatasetDict:
    """Load the houses dataset."""
    data = {}
    for split in ["val", "test"]:
        split_task_list = []
        for task in ["objectnavtype", "fetch2roomtype", "explorehouse"]:
            filename = f"minival_June8_procthor_100k_{task}_{split}.jsonl.gz"
            print(filename)
            try:
                with gzip.open(filename, "r") as f:
                    tasks = [line for line in tqdm(f, desc=f"Loading {split}")]
                split_task_list.extend(tasks)
            except:
                print(f"Error: {task} for {split} not found")
                continue

        data[split] = LazyJsonDataset(
            data=tasks, dataset="procthor-100k-eval", split=split
        )
    return prior.DatasetDict(**data)
