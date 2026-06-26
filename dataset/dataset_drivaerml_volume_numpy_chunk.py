import os
import json
import numpy as np
import torch
import torch.utils.data as data
import random


class DrivAerChunkDataset(data.Dataset):
    def __init__(self, root, data_list, train=True, label_fields=None, chunk_suffix="part",
                 norm_stats_file=None):
        """
        root: chunked data root directory, e.g. /data/DrivaerML_volume_chunked
        data_list: dict with keys 'train_list' and 'test_list'
        label_fields: list of field names to load, default ['CptMeanTrim', 'UMeanTrim']
        """
        self.root = root
        self.train = train
        self.label_fields = label_fields or ['CptMeanTrim', 'UMeanTrim']
        self.run_list = data_list["train_list"] if train else data_list["test_list"]
        self.chunk_suffix = chunk_suffix

        self.run_chunks = {}

        # Normalization statistics computed over the training set
        # Fields: [CptMeanTrim, UMeanTrim_x, UMeanTrim_y, UMeanTrim_z]
        self.label_mean = np.array([1.71387e-01, 1.67909e01, -3.82238e-02, 4.07968e-01])
        self.label_std = np.array([5.00826e-01, 1.64115e01, 8.63614e00, 6.64996e00])

        exist_run_list = []
        for run_name in self.run_list:
            run_dir = os.path.join(root, run_name)

            if not os.path.exists(run_dir):
                print(f"Warning: run directory not found: {run_dir}")
                continue

            point_chunk_files = sorted([
                f for f in os.listdir(run_dir)
                if f.startswith(f"{run_name}_cell_centers_{chunk_suffix}")
            ])
            exist_run_list.append(run_name)
            parts = []
            if self.train:
                for f_name in point_chunk_files:
                    prefix = f"{run_name}_cell_centers_"
                    assert f_name.startswith(prefix), f"Unexpected filename format: {f_name}"
                    stem = f_name[len(prefix):-4]  # e.g. "part3"
                    parts.append(stem)
            else:
                for f_name in point_chunk_files[:20]:
                    prefix = f"{run_name}_cell_centers_"
                    assert f_name.startswith(prefix), f"Unexpected filename format: {f_name}"
                    stem = f_name[len(prefix):-4]
                    parts.append(stem)
            self.run_chunks[run_name] = parts

        self.run_list = exist_run_list
        print(f"ChunkDataset initialized with {len(self.run_list)} runs")

    def __len__(self):
        if self.train:
            return len(self.run_list)
        else:
            return sum(len(parts) for parts in self.run_chunks.values())

    def __getitem__(self, idx):
        if self.train:
            run_name = self.run_list[idx]

            # Randomly select one chunk part for this run
            parts = self.run_chunks[run_name]
            part_id = random.choice(parts)

            run_dir = os.path.join(self.root, run_name)

            # Load cell center coordinates
            points = np.load(
                os.path.join(run_dir, f"{run_name}_cell_centers_{part_id}.npy")
            )

            # Load label fields and concatenate
            label_arrays = []
            for field in self.label_fields:
                arr = np.load(os.path.join(run_dir, f"{run_name}_{field}_{part_id}.npy"))
                if arr.ndim == 1:
                    arr = arr[:, None]
                label_arrays.append(arr)
            labels = np.concatenate(label_arrays, axis=1)
            labels = (labels - self.label_mean) / (self.label_std + 1e-8)

            # Subsample every 4th point to reduce memory per batch
            x = random.randint(0, 4)
            points = torch.from_numpy(points[x::4]).float()
            labels = torch.from_numpy(labels[x::4]).float()

            return [points], [labels], [points], run_name

        else:
            num_parts = len(self.run_chunks[self.run_list[0]])
            run_name = self.run_list[idx // num_parts]
            parts = self.run_chunks[run_name]
            parts = sorted(parts, key=lambda x: int(x.replace('part', '')))
            part_id = parts[idx % num_parts]

            run_dir = os.path.join(self.root, run_name)

            points = np.load(
                os.path.join(run_dir, f"{run_name}_cell_centers_{part_id}.npy")
            )

            label_arrays = []
            for field in self.label_fields:
                arr = np.load(os.path.join(run_dir, f"{run_name}_{field}_{part_id}.npy"))
                if arr.ndim == 1:
                    arr = arr[:, None]
                label_arrays.append(arr)
            labels = np.concatenate(label_arrays, axis=1)
            labels = (labels - self.label_mean) / (self.label_std + 1e-8)

            points = torch.from_numpy(points).float()
            labels = torch.from_numpy(labels).float()

            return points, labels, points, run_name


class DrivAerMLVTUChunkDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, sampler=None, num_workers=16):
        super().__init__(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
        self.run_list = dataset.run_list
