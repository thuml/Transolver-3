import os
import json
import numpy as np
import torch
import torch.utils.data as data
import random


class DrivAerChunkDataset(data.Dataset):
    def __init__(self, root, data_list, train=True, label_fields=None, chunk_suffix="part", norm_stats_file=None):
        """
        root: chunk 数据根目录，例如 /data/DrivAerML_chunks
        run_list: 所有 run_xxx 名称
        label_fields: ['nutMeanTrim', 'pMeanTrim', 'UMeanTrim']
        """
        self.root = root
        self.train = train
        self.label_fields = label_fields or ['CptMeanTrim', 'UMeanTrim']
        self.run_list = data_list["train_list"] if train else data_list["test_list"]
        self.chunk_suffix = chunk_suffix

        self.run_chunks = {}

        self.label_mean = np.array([1.71387e-01, 1.67909e01, -3.82238e-02, 4.07968e-01])
        self.label_std = np.array([5.00826e-01, 1.64115e01, 8.63614e00, 6.64996e00])
        exist_run_list = []
        for run_name in self.run_list:
            run_dir = os.path.join(root, run_name)

            if not os.path.exists(run_dir):
                print(f"❌ 没有找到 chunk 文件: {run_dir}")
                pass
            else:
                point_chunk_files = sorted([
                    f for f in os.listdir(run_dir)
                    if f.startswith(f"{run_name}_cell_centers_{chunk_suffix}")
                ])
                exist_run_list.append(run_name)
                parts = []
                if self.train:
                    for f_name in point_chunk_files:
                        # f_name 形如 "run_001_points_part3.npy"
                        # 去掉前缀 "run_001_points_" 和后缀 ".npy"
                        prefix = f"{run_name}_cell_centers_"
                        assert f_name.startswith(prefix), f"文件名格式不对: {f_name}"
                        stem = f_name[len(prefix):-4]      # "part3"
                        parts.append(stem)
                else:
                    for f_name in point_chunk_files[:1]:
                        # f_name 形如 "run_001_points_part3.npy"
                        # 去掉前缀 "run_001_points_" 和后缀 ".npy"
                        prefix = f"{run_name}_cell_centers_"
                        assert f_name.startswith(prefix), f"文件名格式不对: {f_name}"
                        stem = f_name[len(prefix):-4]      # "part3"
                        parts.append(stem)
                self.run_chunks[run_name] = parts
        self.run_list = exist_run_list
        print(f"✔ ChunkDataset 初始化完成，共 {len(self.run_list)} runs")

    def __len__(self):
        if self.train:
            return len(self.run_list)
        else:
            total_parts = 0
            for run_name in self.run_chunks.keys():
                total_parts += len(self.run_chunks[run_name])
            return total_parts
            # return 4

    def __getitem__(self, idx):
        if self.train:
            run_name = self.run_list[idx]

            # -------- 1. 随机挑选一个 part --------
            parts = self.run_chunks[run_name]
            part_id = random.choice(parts)  # 如 "_part3"

            run_dir = os.path.join(self.root, run_name)

            # -------- 2. 加载 points --------
            points = np.load(
                os.path.join(run_dir, f"{run_name}_cell_centers_{part_id}.npy")
            )

            # -------- 3. 加载 labels（多个字段）--------
            label_arrays = []
            for field in self.label_fields:
                if field == 'CptMeanTrim':
                    tmp_run_dir = os.path.join('/workspace2/zhouhang/driverml/DrivaerML_volume_chunk_interval_400_cpt_no_shuffle', run_name)
                    arr = np.load(os.path.join(tmp_run_dir, f"{run_name}_{field}_{part_id}.npy"))
                else:
                    arr = np.load(os.path.join(run_dir, f"{run_name}_{field}_{part_id}.npy"))
                if len(arr.shape)==1:
                    arr = arr[:, None]
                label_arrays.append(arr)
            labels = np.concatenate(label_arrays, axis=1)
            labels = (labels - self.label_mean) / (self.label_std + 1e-8)
            x = random.randint(0, 9)
            # -------- 4. 返回 torch tensor --------
            points = torch.from_numpy(points[x::10]).float()
            labels = torch.from_numpy(labels[x::10]).float()
            print(points.shape)
            input()
            return [points], [labels], [points], run_name
        else:
            # run_name = self.run_list[idx // 400]
            run_name = self.run_list[idx//1]
            # -------- 1. 随机挑选一个 part --------
            parts = self.run_chunks[run_name]
            parts = sorted(parts, key=lambda x: int(x.replace('part', '')))
            # part_id = parts[idx % 400]  # 如 "_part3"
            part_id = parts[idx%1]
            run_dir = os.path.join(self.root, run_name)

            # -------- 2. 加载 points --------
            points = np.load(
                os.path.join(run_dir, f"{run_name}_cell_centers_{part_id}.npy")
            )

            # -------- 3. 加载 labels（多个字段）--------
            label_arrays = []
            for field in self.label_fields:
                if field == 'CptMeanTrim':
                    tmp_run_dir = os.path.join('/workspace2/zhouhang/driverml/DrivaerML_volume_chunk_interval_400_cpt_no_shuffle', run_name)
                    arr = np.load(os.path.join(tmp_run_dir, f"{run_name}_{field}_{part_id}.npy"))
                else:
                    arr = np.load(os.path.join(run_dir, f"{run_name}_{field}_{part_id}.npy"))
                if len(arr.shape)==1:
                    arr = arr[:, None]
                label_arrays.append(arr)
            labels = np.concatenate(label_arrays, axis=1)
            labels = (labels - self.label_mean) / (self.label_std + 1e-8)
            # -------- 4. 返回 torch tensor --------
            points = torch.from_numpy(points[::1]).float()
            labels = torch.from_numpy(labels[::1]).float()

            return points, labels, points, run_name


class DrivAerMLVTUChunkDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, sampler=None, num_workers=16):
        super().__init__(dataset, batch_size=batch_size, sampler=sampler, num_workers=16)
        self.run_list = dataset.run_list