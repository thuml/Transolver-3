import os
import json
import numpy as np
import torch
import torch.utils.data as data
import random
import pickle


class DrivAerChunkDataset(data.Dataset):
    def __init__(self, root, data_list, train=True, label_fields=None, chunk_suffix="part", norm_stats_file=None):
        """
        root: chunked data root directory, e.g. /data/DrivAerML_surface_chunks
        data_list: dict with 'train_list' and 'test_list' keys (loaded from JSON split file)
        label_fields: list of field names to load as labels; defaults to ['pMeanTrim', 'wallShearStressMeanTrim']
        chunk_suffix: prefix for chunk filename stems (e.g. "part" -> "boundary_001_points_part0.npy")
        norm_stats_file: optional path to a .pkl file with 'label_mean' and 'label_std' arrays;
                         if None, uses pre-computed statistics from the DrivAerML training set
        """
        self.root = root
        self.label_fields = label_fields if label_fields is not None else ['pMeanTrim', 'wallShearStressMeanTrim']
        self.run_list = data_list["train_list"] if train else data_list["test_list"]
        self.chunk_suffix = chunk_suffix
        self.train = train
        self.run_chunks = {}

        # Normalization statistics computed on the DrivAerML training set
        if norm_stats_file is not None:
            with open(norm_stats_file, 'rb') as f:
                stats = pickle.load(f)
            self.label_mean = np.array(stats['label_mean'])
            self.label_std = np.array(stats['label_std'])
        else:
            self.label_mean = np.array([-2.30207226e+02, -1.20971349e+00, 1.44910027e-03, -7.12132631e-02])
            self.label_std = np.array([2.68560778e+02, 2.07625744e+00, 1.35203571e+00, 1.10551982e+00])

        for run_name in self.run_list:
            run_dir = os.path.join(root, run_name)
            boundary_name = run_name.replace('run', 'boundary')
            point_chunk_files = sorted([
                f for f in os.listdir(run_dir)
                if f.startswith(f"{boundary_name}_points_{chunk_suffix}")
            ])
            if len(point_chunk_files) == 0:
                raise ValueError(f"No chunk files found in: {run_dir}")

            parts = []
            if self.train:
                for f_name in point_chunk_files:
                    prefix = f"{boundary_name}_points_"
                    assert f_name.startswith(prefix), f"Unexpected filename format: {f_name}"
                    stem = f_name[len(prefix):-4]
                    parts.append(stem)
                self.run_chunks[run_name] = parts
            else:
                for f_name in point_chunk_files:
                    prefix = f"{boundary_name}_points_"
                    assert f_name.startswith(prefix), f"Unexpected filename format: {f_name}"
                    stem = f_name[len(prefix):-4]
                    parts.append(stem)
                self.run_chunks[run_name] = parts

        print(f"ChunkDataset initialized with {len(self.run_list)} runs")

    def __len__(self):
        if self.train:
            return len(self.run_list)
        else:
            total_parts = 0
            for run_name in self.run_chunks.keys():
                total_parts += len(self.run_chunks[run_name])
            return total_parts

    def __getitem__(self, idx):
        if self.train:
            run_name = self.run_list[idx]
            boundary_name = run_name.replace('run', 'boundary')
            parts = self.run_chunks[run_name]
            part_id = random.choice(parts)

            run_dir = os.path.join(self.root, run_name)

            points = np.load(os.path.join(run_dir, f"{boundary_name}_points_{part_id}.npy"))
            normals = np.load(os.path.join(run_dir, f"{boundary_name}_normals_{part_id}.npy"))

            label_arrays = []
            for field in self.label_fields:
                arr = np.load(os.path.join(run_dir, f"{boundary_name}_{field}_{part_id}.npy"))
                if len(arr.shape) == 1:
                    arr = arr[:, None]
                label_arrays.append(arr)
            labels = np.concatenate(label_arrays, axis=1)
            labels = (labels - self.label_mean) / (self.label_std + 1e-10)

            x = random.randint(0, 3)
            points_chunk = torch.from_numpy(points[x::4]).float()
            normals_chunk = torch.from_numpy(normals[x::4]).float()
            labels_chunk = torch.from_numpy(labels[x::4]).float()
            points_list = [torch.concat([points_chunk, normals_chunk], dim=-1)]
            labels_list = [labels_chunk]

            return points_list, labels_list, points_list, run_name
        else:
            num_parts = len(self.run_chunks[self.run_list[0]])
            run_name = self.run_list[idx // num_parts]
            boundary_name = run_name.replace('run', 'boundary')
            parts = self.run_chunks[run_name]
            parts = sorted(parts, key=lambda x: int(x.replace('part', '')))
            part_id = parts[idx % num_parts]

            run_dir = os.path.join(self.root, run_name)

            points = np.load(os.path.join(run_dir, f"{boundary_name}_points_{part_id}.npy"))
            normals = np.load(os.path.join(run_dir, f"{boundary_name}_normals_{part_id}.npy"))

            label_arrays = []
            for field in self.label_fields:
                arr = np.load(os.path.join(run_dir, f"{boundary_name}_{field}_{part_id}.npy"))
                if len(arr.shape) == 1:
                    arr = arr[:, None]
                label_arrays.append(arr)
            labels = np.concatenate(label_arrays, axis=1)
            labels = (labels - self.label_mean) / (self.label_std + 1e-10)

            points = torch.from_numpy(points).float()
            normals = torch.from_numpy(normals).float()
            points = torch.concat([points, normals], dim=-1)
            labels = torch.from_numpy(labels).float()

            return points, labels, points, run_name


class DrivAerMLVTUChunkDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, sampler=None, num_workers=8):
        super().__init__(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
        self.run_list = dataset.run_list
