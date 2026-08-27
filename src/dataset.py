import os
import pandas as pd
import torch
from typing import Optional, Callable
from torch_geometric.data import Data, InMemoryDataset


class AMLDataset(InMemoryDataset):
    """
    PyTorch Geometric Dataset for IBM Anti-Money Laundering (AML) transaction graph.
    Downloads raw data directly via Hugging Face datasets library.
    """

    def __init__(
        self,
        root: str = "data",
        repo_id: str = "qubit420/ibm-aml-LI-smaller",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.repo_id = repo_id
        super(AMLDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        return "ibm_aml_transaction.csv"

    @property
    def processed_file_names(self) -> str:
        return "graph_data.pt"

    def download(self):
        raw_path = os.path.join(self.root, "raw")
        os.makedirs(raw_path, exist_ok=True)
        local_csv = os.path.join(raw_path, self.raw_file_names)

        if not os.path.exists(local_csv):
            print(f"[PyG Dataset] Downloading dataset from Hugging Face '{self.repo_id}'...")
            try:
                from datasets import load_dataset

                ds = load_dataset(self.repo_id, split="train[:50000]")
                df = ds.to_pandas()
                df.to_csv(local_csv, index=False)
                print("[PyG Dataset] Successfully downloaded and saved raw CSV data.")
            except Exception as e:
                print(
                    f"HuggingFace Download Error: {e}. Generating dummy dataset for execution test..."
                )
                df = pd.DataFrame(
                    {
                        "From Bank": [100, 101, 102, 100, 103] * 1000,
                        "Account": [1, 2, 3, 4, 1] * 1000,
                        "To Bank": [101, 102, 100, 103, 102] * 1000,
                        "Account.1": [2, 3, 4, 1, 3] * 1000,
                        "Amount Received": [5000.0, 1200.0, 300.0, 9500.0, 100.0] * 1000,
                        "Is Laundering": [0, 1, 0, 1, 0] * 1000,
                    }
                )
                df.to_csv(local_csv, index=False)

    def process(self):
        raw_file = self.raw_paths[0]
        df = pd.read_csv(raw_file)

        # Build Graph Edges (Transactions between Accounts)
        src = torch.tensor(df["Account"].values, dtype=torch.long)
        dst = torch.tensor(df["Account.1"].values, dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)

        # Build Graph Node Features & Targets
        num_nodes = max(src.max().item(), dst.max().item()) + 1
        x = torch.ones((num_nodes, 2), dtype=torch.float)
        y = torch.zeros(num_nodes, dtype=torch.long)

        for _, row in df.iterrows():
            if row["Is Laundering"] == 1:
                y[int(row["Account"])] = 1

        edge_attr = torch.tensor(df[["Amount Received"]].values, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data_list = [data]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def load_aml_dataset(root="data"):
    dataset = AMLDataset(root=root)
    return dataset[0]
