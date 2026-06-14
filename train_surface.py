import numpy as np
import time, json, os
import torch
import torch.nn as nn
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data.distributed import DistributedSampler

def get_nb_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

def train(device, model, train_loader, optimizer, scheduler,
          reg=1, pos_norm=0, norm_norm=0, out_norm=1,
          pos_mean=None, pos_std=None, norm_mean=None, norm_std=None, out_mean=None, out_std=None,
          epoch_num=0, ema_slice_tokens={}):
    model.train()
    losses_mse = []
    lr = optimizer.param_groups[0]['lr']
    print(lr)
    for batch_idx, (x, y, pos, run_name) in enumerate(train_loader):
        optimizer.zero_grad()

        x_list = [xi.to(device) for xi in x]
        y_list = [yi.to(device) for yi in y]
        pos_list = [pi.to(device) for pi in pos]

        data_list = list(zip(x_list, pos_list))  # [(x1,pos1), (x2,pos2), ...]

        # 你的 chunked 前向：返回 list[(B,ni,out_dim)]
        out_list = model(data_list)

        # 计算“所有 chunk 上的全元素平均 MSE”（等价于把点拼起来算 mean）
        sq_sum = 0.0
        elem_cnt = 0
        for out_k, y_k in zip(out_list, y_list):
            diff = out_k - y_k
            sq_sum = sq_sum + (diff * diff).sum()
            elem_cnt += diff.numel()

        loss_press = sq_sum / max(elem_cnt, 1)
        loss_press.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses_mse.append(loss_press.item())

    scheduler.step()
    return float(np.mean(losses_mse))

@torch.no_grad()
def test(device, model, test_loader, pos_norm=0, norm_norm=1, out_norm=1, pos_mean=None, pos_std=None, norm_mean=None, norm_std=None, out_mean=None, out_std=None, full=False):
    model.eval()

    criterion_func_mse = nn.MSELoss(reduction='none')
    criterion_func_mae = nn.L1Loss(reduction='none')
    
    losses_mse = []
    losses_mae = []
    losses_l2re = []
    step = 0
    result_pos = []
    result_predict = {}
    # result_gt = []

    for batch_idx, (x, y, pos, run_name) in enumerate(test_loader):
        step += 1
        x = x.to(device)
        pos = pos.to(device)
        result_pos.append(pos.detach().cpu().numpy()[0])
        if pos_norm:
            pos = (pos - pos_mean) / pos_std
            x[:, :, :3] = pos
        if norm_norm:
            norm = (x[:, :, 4:] - norm_mean) / norm_std
            x[:, :, 4:] = norm
        y = y.to(device)
        out = model([(x, pos)])[0]
        if run_name not in result_predict.keys():
            result_predict[run_name] = []
        result_predict[run_name].append(out.detach().cpu().numpy()[0])
        
        
        # result_gt.append(y.detach().cpu().numpy()[0])
        if out_norm:
            y_norm = (y - out_mean) / (out_std + 1e-6) # normalize label
            loss_mse_elementwise = criterion_func_mse(out, y_norm)
            loss_mse_per_feature_mean = torch.mean(loss_mse_elementwise, dim=(0, 1))

            loss_mae_elementwise = criterion_func_mae(out, y_norm)
            loss_mae_per_feature_mean = torch.mean(loss_mae_elementwise, dim=(0, 1))

            out = out * out_std + out_mean # denormalize output
            diff = y - out
            l2_norm_per_feature = torch.norm(diff, p=2, dim=[0, 1])
            relative_l2_error_per_feature = l2_norm_per_feature / torch.norm(y, p=2, dim=[0, 1])
        else:
            loss_mse_elementwise = criterion_func_mse(out, y)
            loss_mse_per_feature_mean = torch.mean(loss_mse_elementwise, dim=(0, 1))

            loss_mae_elementwise = criterion_func_mae(out, y)
            loss_mae_per_feature_mean = torch.mean(loss_mae_elementwise, dim=(0, 1))
            
            label_mean = torch.from_numpy(np.array([-2.30207226e+02, -1.20971349e+00, 1.44910027e-03, -7.12132631e-02]))[None, None, :].cuda().float()
            label_std = torch.from_numpy(np.array([2.68560778e+02, 2.07625744e+00, 1.35203571e+00, 1.10551982e+00]))[None, None, :].cuda().float()
            y = y * label_std + label_mean
            out = out * label_std + label_mean
            
            y_vel = y[..., -3:]
            out_vel = out[..., -3:]

            y_speed = torch.norm(y_vel, p=2, dim=-1)[:, :, None]
            out_speed = torch.norm(out_vel, p=2, dim=-1)[:, :, None]

            y = torch.concatenate([y[:, :, :1], y_speed], dim=-1)
            out = torch.concatenate([out[:, :, :1], out_speed], dim=-1)
            
            diff = y - out
            l2_norm_per_feature = torch.norm(diff, p=2, dim=[0, 1])
            relative_l2_error_per_feature = l2_norm_per_feature / torch.norm(y, p=2, dim=[0, 1])

        losses_mse.append(loss_mse_per_feature_mean.cpu().numpy())
        losses_mae.append(loss_mae_per_feature_mean.cpu().numpy())
        losses_l2re.append(relative_l2_error_per_feature.cpu().numpy())
    
    # for run_name in result_predict.keys():
    #     N = sum(batch.shape[0] for batch in result_predict[run_name])   # 全部点数
    #     F = result_predict[run_name][0].shape[1]
    #     print(N, F)
    #     num_batches = 20

    #     result_predict_full = np.zeros((N, F))
    #     result_pos_full = np.zeros((N, 6))
    #     result_gt_full = np.zeros((N, F))
        
    #     for b in range(num_batches):
    #         temp_tensor = np.zeros_like(result_predict_full[b::num_batches])
    #         for idx in range(1):
    #             temp_tensor[idx::1] = result_predict[run_name][b * 1 + idx]
    #         result_predict_full[b::num_batches] = temp_tensor

    #     np.save(f"/workspace2/zhouhang/DrivAerML_prediction_results/predict_sorted_no_amortize_24layer_100000_800epoch_{run_name[0]}.npy", result_predict_full)
    #     print('saved')
    
    return np.mean(losses_mse, axis=0), np.mean(losses_mae, axis=0), np.mean(losses_l2re, axis=0)

@torch.no_grad()
def test_decoupled_inference(device, model1, model2, test_loader1, test_loader2,
                             max_ref_chunks=None):
    """
    Decoupled inference framework (Transolver-3, Section 3.3).

    Stage 1 - Physical state caching: iterate test_loader1 layer-by-layer to
    build the physical state cache s_cache for each run. max_ref_chunks caps
    how many chunks per run are used during caching (None = all chunks).

    Stage 2 - Full mesh decoding: run test_loader2 using the physical state cache.

    Returns: (mean_mse, mean_mae, mean_l2re) averaged over all query batches.
    """
    model1.eval()
    model2.eval()
    criterion_func_mse = nn.MSELoss(reduction='none')
    criterion_func_mae = nn.L1Loss(reduction='none')

    losses_mse = []
    losses_mae = []
    losses_l2re = []

    # Normalization stats: [pressure, wall-shear_x, wall-shear_y, wall-shear_z]
    label_mean = torch.tensor(
        [-2.30207226e+02, -1.20971349e+00, 1.44910027e-03, -7.12132631e-02],
        device=device, dtype=torch.float
    )[None, None, :]
    label_std = torch.tensor(
        [2.68560778e+02, 2.07625744e+00, 1.35203571e+00, 1.10551982e+00],
        device=device, dtype=torch.float
    )[None, None, :]

    n_layers = len(model1.module.blocks if hasattr(model1, 'module') else model1.blocks)

    # ---- Stage 1: physical state caching ----
    state_cache_all = {}  # run_name -> physical state cache (one state per layer)
    t0 = time.time()

    for layer in range(n_layers):
        state_num_layer = {}   # run_name -> accumulated unnormalized numerator
        state_den_layer = {}   # run_name -> accumulated denominator

        chunks_seen = {}
        for batch_idx, (x, y, pos, run_name) in tqdm(enumerate(test_loader1),
                                                      desc=f'caching layer {layer}'):
            if max_ref_chunks is not None:
                chunks_seen.setdefault(run_name, 0)
                if chunks_seen[run_name] >= max_ref_chunks:
                    continue
                chunks_seen[run_name] += 1

            if run_name not in state_cache_all:
                state_cache_all[run_name] = []
            if run_name not in state_num_layer:
                state_num_layer[run_name] = 0
                state_den_layer[run_name] = 0

            x = x.to(device)
            pos = pos.to(device)
            _, slice_token_wo_norm, slice_norm = model1(
                [(x, pos)], state_cache_all[run_name], layer
            )
            state_num_layer[run_name] = state_num_layer[run_name] + slice_token_wo_norm
            state_den_layer[run_name] = state_den_layer[run_name] + slice_norm

        # Normalize and append to each run's physical state cache
        for run_name in state_num_layer:
            norm = (state_den_layer[run_name] + 1e-5)[..., None]
            normalized = state_num_layer[run_name] / norm
            state_cache_all[run_name].append(normalized)

    print(f'Physical state caching: {time.time() - t0:.1f}s')

    # ---- Stage 2: full mesh decoding ----
    t1 = time.time()
    for batch_idx, (x, y, pos, run_name) in enumerate(test_loader2):
        x = x.to(device)
        pos = pos.to(device)
        y = y.to(device)

        out = model2([(x, pos)], state_cache_all[run_name])[0]

        loss_mse_per_feature_mean = torch.mean(criterion_func_mse(out, y), dim=(0, 1))
        loss_mae_per_feature_mean = torch.mean(criterion_func_mae(out, y), dim=(0, 1))

        # Denormalize to physical units for L2RE computation
        y_phys = y * label_std + label_mean
        out_phys = out * label_std + label_mean

        # Report L2RE on [pressure, |wall-shear|] (scalar speed instead of 3 components)
        y_speed = torch.norm(y_phys[..., -3:], p=2, dim=-1, keepdim=True)
        out_speed = torch.norm(out_phys[..., -3:], p=2, dim=-1, keepdim=True)
        y_eval = torch.cat([y_phys[..., :1], y_speed], dim=-1)
        out_eval = torch.cat([out_phys[..., :1], out_speed], dim=-1)

        diff = y_eval - out_eval
        relative_l2_error_per_feature = (
            torch.norm(diff, p=2, dim=[0, 1]) / torch.norm(y_eval, p=2, dim=[0, 1])
        )

        losses_mse.append(loss_mse_per_feature_mean.cpu().numpy())
        losses_mae.append(loss_mae_per_feature_mean.cpu().numpy())
        losses_l2re.append(relative_l2_error_per_feature.cpu().numpy())

    print(f'Full mesh decoding: {time.time() - t1:.1f}s  |  Total: {time.time() - t0:.1f}s')
    return np.mean(losses_mse, axis=0), np.mean(losses_mae, axis=0), np.mean(losses_l2re, axis=0)

@torch.no_grad()
def inference(device, model1, model2, test_loader, pos_norm=0, norm_norm=1, out_norm=1, pos_mean=None, pos_std=None, norm_mean=None, norm_std=None, out_mean=None, out_std=None, full=False):
    model1.eval()
    model2.eval()
    criterion_func_mse = nn.MSELoss(reduction='none')
    criterion_func_mae = nn.L1Loss(reduction='none')
    
    losses_mse = []
    losses_mae = []
    losses_l2re = []
    step = 0
    result_pos = []
    result_predict = []
    result_gt = []

    slice_token_wo_norm_all = []
    slice_norm_all = []
    slice_aggregate_all = []
    
    # for batch_idx, (x, y, pos, run_name) in enumerate(test_loader):
    #     step += 1
    #     x = x.to(device)
    #     pos = pos.to(device)
    #     if pos_norm:
    #         pos = (pos - pos_mean) / pos_std
    #         x[:, :, :3] = pos
    #     if norm_norm:
    #         norm = (x[:, :, 4:] - norm_mean) / norm_std
    #         x[:, :, 4:] = norm
    #     y = y.to(device)
    #     out, slice_token_wo_norm, slice_norm = model1((x, pos), None)
    #     slice_token_wo_norm_all += slice_token_wo_norm
    #     slice_norm_all += slice_norm
    # slice_token_wo_norm_all = slice_token_wo_norm_all \
    #                                 /((slice_norm_all + 1e-5)[:, :, :, :, None].repeat(1, 1, 1, 1, slice_token_wo_norm_all.shape[-1]))

    # slice_aggregate_all = torch.stack(slice_aggregate_all)
    for batch_idx, (x, y, pos, run_name) in enumerate(test_loader):
        step += 1
        x = x.to(device)
        pos = pos.to(device)
        result_pos.append(pos.detach().cpu().numpy()[0])
        if pos_norm:
            print('pos_norm')
            input()
            pos = (pos - pos_mean) / pos_std
            x[:, :, :3] = pos
        if norm_norm:
            print('norm_norm')
            input()
            norm = (x[:, :, 4:] - norm_mean) / norm_std
            x[:, :, 4:] = norm
        y = y.to(device)
        out, _ = model2((x, pos), None, 0)
        result_predict.append(out.detach().cpu().numpy()[0])
        result_gt.append(y.detach().cpu().numpy()[0])
        if out_norm:
            y_norm = (y - out_mean) / (out_std + 1e-6) # normalize label
            loss_mse_elementwise = criterion_func_mse(out, y_norm)
            loss_mse_per_feature_mean = torch.mean(loss_mse_elementwise, dim=(0, 1))

            loss_mae_elementwise = criterion_func_mae(out, y_norm)
            loss_mae_per_feature_mean = torch.mean(loss_mae_elementwise, dim=(0, 1))

            out = out * out_std + out_mean # denormalize output
            diff = y - out
            l2_norm_per_feature = torch.norm(diff, p=2, dim=[0, 1])
            relative_l2_error_per_feature = l2_norm_per_feature / torch.norm(y, p=2, dim=[0, 1])
        else:
            loss_mse_elementwise = criterion_func_mse(out, y)
            loss_mse_per_feature_mean = torch.mean(loss_mse_elementwise, dim=(0, 1))

            loss_mae_elementwise = criterion_func_mae(out, y)
            loss_mae_per_feature_mean = torch.mean(loss_mae_elementwise, dim=(0, 1))

            diff = y - out
            l2_norm_per_feature = torch.norm(diff, p=2, dim=[0, 1])
            relative_l2_error_per_feature = l2_norm_per_feature / torch.norm(y, p=2, dim=[0, 1])

        losses_mse.append(loss_mse_per_feature_mean.cpu().numpy())
        losses_mae.append(loss_mae_per_feature_mean.cpu().numpy())
        losses_l2re.append(relative_l2_error_per_feature.cpu().numpy())
    
    N = sum(batch.shape[0] for batch in result_predict)   # 全部点数
    F = result_predict[0].shape[1]
    print(N, F)
    num_batches = 20

    result_predict_full = np.zeros((N, F))
    
    for b in range(num_batches):
        temp_tensor = np.zeros_like(result_predict_full[b::num_batches])
        for idx in range(4):
            temp_tensor[idx::4] = result_predict[b * 4 + idx]
        result_predict_full[b::num_batches] = temp_tensor

    np.save("predict_sorted_no_amortize.npy", result_predict_full)


    return np.mean(losses_mse, axis=0), np.mean(losses_mae, axis=0), np.mean(losses_l2re, axis=0)

@torch.no_grad()
def inference_amortize(device, model, test_loader, pos_norm=0, norm_norm=1, out_norm=1, pos_mean=None, pos_std=None, norm_mean=None, norm_std=None, out_mean=None, out_std=None, full=False):
    model.eval()
    criterion_func_mse = nn.MSELoss(reduction='none')
    criterion_func_mae = nn.L1Loss(reduction='none')
    
    losses_mse = []
    losses_mae = []
    losses_l2re = []
    step = 0
    hidden_states = [0 for _ in range(8)]
    result_pos = []
    result_predict = []
    result_gt = []
    
    for batch_idx, (x, y, pos, run_name) in enumerate(test_loader):
        step += 1
        x = x.to(device)
        pos = pos.to(device)
        if pos_norm:
            pos = (pos - pos_mean) / pos_std
            x[:, :, :3] = pos
        if norm_norm:
            norm = (x[:, :, 4:] - norm_mean) / norm_std
            x[:, :, 4:] = norm
        y = y.to(device)
        out, hidden_states = model((x, pos), hidden_states, step)

    for batch_idx, (x, y, pos, run_name) in enumerate(test_loader):
        step += 1
        x = x.to(device)
        pos = pos.to(device)
        result_pos.append(pos.detach().cpu().numpy()[0])
        if pos_norm:
            pos = (pos - pos_mean) / pos_std
            x[:, :, :3] = pos
        if norm_norm:
            norm = (x[:, :, 4:] - norm_mean) / norm_std
            x[:, :, 4:] = norm
        y = y.to(device)
        out, hidden_states = model((x, pos), hidden_states, step)
        result_predict.append(out.detach().cpu().numpy()[0])
        result_gt.append(y.detach().cpu().numpy()[0])
        if out_norm:
            y_norm = (y - out_mean) / (out_std + 1e-6) # normalize label
            loss_mse_elementwise = criterion_func_mse(out, y_norm)
            loss_mse_per_feature_mean = torch.mean(loss_mse_elementwise, dim=(0, 1))

            loss_mae_elementwise = criterion_func_mae(out, y_norm)
            loss_mae_per_feature_mean = torch.mean(loss_mae_elementwise, dim=(0, 1))

            out = out * out_std + out_mean # denormalize output
            diff = y - out
            l2_norm_per_feature = torch.norm(diff, p=2, dim=[0, 1])
            relative_l2_error_per_feature = l2_norm_per_feature / torch.norm(y, p=2, dim=[0, 1])
        else:
            loss_mse_elementwise = criterion_func_mse(out, y)
            loss_mse_per_feature_mean = torch.mean(loss_mse_elementwise, dim=(0, 1))

            loss_mae_elementwise = criterion_func_mae(out, y)
            loss_mae_per_feature_mean = torch.mean(loss_mae_elementwise, dim=(0, 1))

            diff = y - out
            l2_norm_per_feature = torch.norm(diff, p=2, dim=[0, 1])
            relative_l2_error_per_feature = l2_norm_per_feature / torch.norm(y, p=2, dim=[0, 1])

        losses_mse.append(loss_mse_per_feature_mean.cpu().numpy())
        losses_mae.append(loss_mae_per_feature_mean.cpu().numpy())
        losses_l2re.append(relative_l2_error_per_feature.cpu().numpy())

    return np.mean(losses_mse, axis=0), np.mean(losses_mae, axis=0), np.mean(losses_l2re, axis=0)

@torch.no_grad()
def inference_amortize_strict(device, model1, model2, test_loader, pos_norm=0, norm_norm=1, out_norm=1, pos_mean=None, pos_std=None, norm_mean=None, norm_std=None, out_mean=None, out_std=None, full=False):
    model1.eval()
    model2.eval()
    criterion_func_mse = nn.MSELoss(reduction='none')
    criterion_func_mae = nn.L1Loss(reduction='none')
    
    losses_mse = []
    losses_mae = []
    losses_l2re = []
    step = 0
    result_pos = []
    result_predict = []
    result_gt = []

    slice_token_wo_norm_all = []
    slice_norm_all = []
    slice_aggregate_all = []
    for layer in range(0, 8):
        slice_token_wo_norm_current_layer = 0
        slice_norm_current_layer = 0
        for batch_idx, (x, y, pos, run_name) in enumerate(test_loader):
            step += 1
            x = x.to(device)
            pos = pos.to(device)
            # if pos_norm:
            #     pos = (pos - pos_mean) / pos_std
            #     x[:, :, :3] = pos
            # if norm_norm:
            #     norm = (x[:, :, 4:] - norm_mean) / norm_std
            #     x[:, :, 4:] = norm
            y = y.to(device)
            out, slice_token_wo_norm, slice_norm = model1((x, pos), slice_aggregate_all, layer)
            slice_token_wo_norm_current_layer += slice_token_wo_norm
            slice_norm_current_layer += slice_norm
        slice_token_wo_norm_current_layer = slice_token_wo_norm_current_layer \
                                        /((slice_norm_current_layer + 1e-5)[:, :, :, None].repeat(1, 1, 1, slice_token_wo_norm_current_layer.shape[-1]))
        slice_aggregate_all.append(slice_token_wo_norm_current_layer)

    slice_aggregate_all = torch.stack(slice_aggregate_all)
    for batch_idx, (x, y, pos, run_name) in enumerate(test_loader):
        step += 1
        x = x.to(device)
        pos = pos.to(device)
        result_pos.append(pos.detach().cpu().numpy()[0])
        if pos_norm:
            pos = (pos - pos_mean) / pos_std
            x[:, :, :3] = pos
        if norm_norm:
            norm = (x[:, :, 4:] - norm_mean) / norm_std
            x[:, :, 4:] = norm
        y = y.to(device)
        out, _ = model2((x, pos), slice_aggregate_all)
        result_predict.append(out.detach().cpu().numpy()[0])
        result_gt.append(y.detach().cpu().numpy()[0])
        if out_norm:
            y_norm = (y - out_mean) / (out_std + 1e-6) # normalize label
            loss_mse_elementwise = criterion_func_mse(out, y_norm)
            loss_mse_per_feature_mean = torch.mean(loss_mse_elementwise, dim=(0, 1))

            loss_mae_elementwise = criterion_func_mae(out, y_norm)
            loss_mae_per_feature_mean = torch.mean(loss_mae_elementwise, dim=(0, 1))

            out = out * out_std + out_mean # denormalize output
            diff = y - out
            l2_norm_per_feature = torch.norm(diff, p=2, dim=[0, 1])
            relative_l2_error_per_feature = l2_norm_per_feature / torch.norm(y, p=2, dim=[0, 1])
        else:
            loss_mse_elementwise = criterion_func_mse(out, y)
            loss_mse_per_feature_mean = torch.mean(loss_mse_elementwise, dim=(0, 1))

            loss_mae_elementwise = criterion_func_mae(out, y)
            loss_mae_per_feature_mean = torch.mean(loss_mae_elementwise, dim=(0, 1))

            diff = y - out
            l2_norm_per_feature = torch.norm(diff, p=2, dim=[0, 1])
            relative_l2_error_per_feature = l2_norm_per_feature / torch.norm(y, p=2, dim=[0, 1])

        losses_mse.append(loss_mse_per_feature_mean.cpu().numpy())
        losses_mae.append(loss_mae_per_feature_mean.cpu().numpy())
        losses_l2re.append(relative_l2_error_per_feature.cpu().numpy())
        
    # N = sum(batch.shape[0] for batch in result_predict)   # 全部点数
    # F = result_predict[0].shape[1]
    # print(N, F)
    # num_batches = 20

    # result_predict_full = np.zeros((N, F))
    # result_pos_full = np.zeros((N, 6))
    # result_gt_full = np.zeros((N, F))
    
    # for b in range(num_batches):
    #     temp_tensor = np.zeros_like(result_predict_full[b::num_batches])
    #     for idx in range(8):
    #         temp_tensor[idx::8] = result_predict[b * 8 + idx]
    #     result_predict_full[b::num_batches] = temp_tensor

    # np.save("predict_sorted_amortize.npy", result_predict_full)

    return np.mean(losses_mse, axis=0), np.mean(losses_mae, axis=0), np.mean(losses_l2re, axis=0)


@torch.no_grad()
def inference_amortize_stl(device, model1, model2, stl_loader, test_loader, pos_norm=0, norm_norm=1, out_norm=1, pos_mean=None, pos_std=None, norm_mean=None, norm_std=None, out_mean=None, out_std=None, full=False):
    model1.eval()
    model2.eval()
    criterion_func_mse = nn.MSELoss(reduction='none')
    criterion_func_mae = nn.L1Loss(reduction='none')
    
    losses_mse = []
    losses_mae = []
    losses_l2re = []
    step = 0
    result_pos = []
    result_predict = []
    result_gt = []

    slice_token_wo_norm_all = []
    slice_norm_all = []
    slice_aggregate_all = []
    for layer in range(0, 8):
        slice_token_wo_norm_current_layer = 0
        slice_norm_current_layer = 0
        for batch_idx, (x, pos) in enumerate(stl_loader):
            step += 1
            x = x.to(device)
            pos = pos.to(device)
            out, slice_token_wo_norm, slice_norm = model1((x, pos), slice_aggregate_all, layer)
            slice_token_wo_norm_current_layer += slice_token_wo_norm
            slice_norm_current_layer += slice_norm
        slice_token_wo_norm_current_layer = slice_token_wo_norm_current_layer \
                                        /((slice_norm_current_layer + 1e-5)[:, :, :, None].repeat(1, 1, 1, slice_token_wo_norm_current_layer.shape[-1]))
        slice_aggregate_all.append(slice_token_wo_norm_current_layer)

    slice_aggregate_all = torch.stack(slice_aggregate_all)
    for batch_idx, (x, y, pos, run_name) in enumerate(test_loader):
        step += 1
        x = x.to(device)
        pos = pos.to(device)
        result_pos.append(pos.detach().cpu().numpy()[0])
        if pos_norm:
            pos = (pos - pos_mean) / pos_std
            x[:, :, :3] = pos
        if norm_norm:
            norm = (x[:, :, 4:] - norm_mean) / norm_std
            x[:, :, 4:] = norm
        y = y.to(device)
        out, _ = model2((x, pos), slice_aggregate_all)
        result_predict.append(out.detach().cpu().numpy()[0])
        result_gt.append(y.detach().cpu().numpy()[0])
        if out_norm:
            y_norm = (y - out_mean) / (out_std + 1e-6) # normalize label
            loss_mse_elementwise = criterion_func_mse(out, y_norm)
            loss_mse_per_feature_mean = torch.mean(loss_mse_elementwise, dim=(0, 1))

            loss_mae_elementwise = criterion_func_mae(out, y_norm)
            loss_mae_per_feature_mean = torch.mean(loss_mae_elementwise, dim=(0, 1))

            out = out * out_std + out_mean # denormalize output
            diff = y - out
            l2_norm_per_feature = torch.norm(diff, p=2, dim=[0, 1])
            relative_l2_error_per_feature = l2_norm_per_feature / torch.norm(y, p=2, dim=[0, 1])
        else:
            loss_mse_elementwise = criterion_func_mse(out, y)
            loss_mse_per_feature_mean = torch.mean(loss_mse_elementwise, dim=(0, 1))

            loss_mae_elementwise = criterion_func_mae(out, y)
            loss_mae_per_feature_mean = torch.mean(loss_mae_elementwise, dim=(0, 1))

            diff = y - out
            l2_norm_per_feature = torch.norm(diff, p=2, dim=[0, 1])
            relative_l2_error_per_feature = l2_norm_per_feature / torch.norm(y, p=2, dim=[0, 1])

        losses_mse.append(loss_mse_per_feature_mean.cpu().numpy())
        losses_mae.append(loss_mae_per_feature_mean.cpu().numpy())
        losses_l2re.append(relative_l2_error_per_feature.cpu().numpy())
        
    # N = sum(batch.shape[0] for batch in result_predict)   # 全部点数
    # F = result_predict[0].shape[1]
    # print(N, F)
    # num_batches = 20

    # result_predict_full = np.zeros((N, F))
    # result_pos_full = np.zeros((N, 6))
    # result_gt_full = np.zeros((N, F))
    
    # for b in range(num_batches):
    #     temp_tensor = np.zeros_like(result_predict_full[b::num_batches])
    #     for idx in range(4):
    #         temp_tensor[idx::4] = result_predict[b * 4 + idx]
    #     result_predict_full[b::num_batches] = temp_tensor

    # np.save("predict_sorted_amortize.npy", result_predict_full)

    return np.mean(losses_mse, axis=0), np.mean(losses_mae, axis=0), np.mean(losses_l2re, axis=0)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_model_state_dict(model):
    if hasattr(model, 'module'):
        model = model.module
    return model.state_dict()

def main(device, train_loader, val_loader, Net, hparams, path, reg=1, val_iter=1, pos_norm=0, out_norm=1, norm_norm=0, pos_mean=None, pos_std=None, out_mean=None, out_std=None, norm_mean=None, norm_std=None, full=False, local_rank=-1):
    model = Net.to(device)
    model = model.float()

    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams['lr']) # 将优化器设置为了AdamW
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hparams['nb_epochs'], eta_min=hparams['lr']*0.01)
    print(lr_scheduler)
    start = time.time()
    ema_slice_tokens = {}
    train_loss, val_loss_mse, val_loss_l2re = 1e5, 1e5, 1e5
    pbar_train = tqdm(range(hparams['nb_epochs']), position=0)
    cnt = 0
    for epoch in pbar_train:
        loss_mse = train(
                device, model, train_loader, optimizer, lr_scheduler, reg,
                pos_norm, norm_norm, out_norm,
                pos_mean, pos_std, norm_mean, norm_std, out_mean, out_std, epoch_num=epoch, ema_slice_tokens=ema_slice_tokens
            )
        train_loss = loss_mse

        if val_iter is not None and (epoch == hparams['nb_epochs'] - 1 or epoch % val_iter == 0):
            loss_mse, loss_mae, loss_l2re = test(device, model, val_loader, pos_norm=pos_norm, out_norm=out_norm, norm_norm=norm_norm, pos_mean=pos_mean, pos_std=pos_std, out_mean=out_mean, out_std=out_std, norm_mean=norm_mean, norm_std=norm_std, full=full)
            val_loss_mse = loss_mse
            val_loss_mae = loss_mae
            val_loss_l2re = loss_l2re

            pbar_train.set_postfix(train_loss=train_loss, val_loss_mse=np.mean(val_loss_mse), val_loss_mae=np.mean(val_loss_mae), val_loss_l2re=np.mean(val_loss_l2re))
            print(f"Epoch {epoch} train loss: {train_loss}, val loss mse: {val_loss_mse}, val loss mae: {val_loss_mae}, val loss l2re: {val_loss_l2re}")
            logging.info(f'Epoch {epoch}, train_loss: {train_loss}, val_loss_mse: {val_loss_mse}, val loss mae: {val_loss_mae}, val_loss_l2re: {val_loss_l2re}')
        else:
            pbar_train.set_postfix(train_loss=train_loss)
            print(f"Epoch {epoch} train loss: {train_loss}")
            logging.info(f'Epoch {epoch}, train_loss: {train_loss}')
        
        if (cnt + 1) % 10 == 0 and local_rank==0:
            torch.save(get_model_state_dict(model), path + os.sep + f'model_{epoch}.pth')
        cnt += 1

    end = time.time()
    time_elapsed = end - start
    params_model = get_nb_trainable_params(model).astype('float')
    print('Number of parameters:', params_model)
    print('Time elapsed: {0:.2f} seconds'.format(time_elapsed))
    logging.info(f'Number of parameters: {params_model}')
    logging.info(f'Time elapsed: {time_elapsed} seconds')
    if local_rank==0:
        torch.save(get_model_state_dict(model), path + os.sep + f'model_{hparams["nb_epochs"]}.pth')

        if val_iter is not None:
            with open(path + os.sep + f'log_{hparams["nb_epochs"]}.json', 'a') as f:
                json.dump(
                    {
                        'nb_parameters': params_model,
                        'time_elapsed': time_elapsed,
                        'hparams': hparams,
                        'train_loss': train_loss,
                        'val_loss_mse': val_loss_mse,
                        'val_loss_l2re': val_loss_l2re,
                    }, f, indent=12, cls=NumpyEncoder
                )

    return model
