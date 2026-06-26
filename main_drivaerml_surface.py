import train_surface
import os
import torch
import argparse
import logging
from dataset.dataset_drivaerml_surface_numpy_chunk import DrivAerChunkDataset, DrivAerMLVTUChunkDataLoader
import json
import numpy as np
from torch.utils.data import SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from models import Transolver_chunk_opt_matrix_mul
from models import Transolver_chunk_opt_matrix_mul_amortize

MODEL_KWARGS = dict(n_hidden=256, n_layers=24, space_dim=6,
                    fun_dim=0, n_head=8, mlp_ratio=2, out_dim=4,
                    slice_num=64, unified_pos=0)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/path/to/DrivAerML_surface_chunked')
parser.add_argument('--val_dir', default=None,
                    help='Validation data directory; defaults to --data_dir if not set')
parser.add_argument('--json_file', default='./drivaerml.json',
                    help='Path to train/test split JSON file')
parser.add_argument('--save_dir', default='./output')
parser.add_argument('--model_ckpt', default='./model.pth',
                    help='Checkpoint path used for --eval mode')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU index used for single-GPU evaluation (ignored during DDP training)')
parser.add_argument('--val_iter', default=10, type=int)
parser.add_argument('--cfd_model', default='Transolver_chunk_opt_matrix_mul', type=str)
parser.add_argument('--r', default=0.2, type=float)
parser.add_argument('--weight', default=0.5, type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--nb_epochs', default=800, type=int)
parser.add_argument('--preprocessed', default=1, type=int)
parser.add_argument('--finetune', default=0, type=int)
parser.add_argument('--pos_norm', default=0, type=int)
parser.add_argument('--norm_norm', default=0, type=int)
parser.add_argument('--out_norm', default=0, type=int)
parser.add_argument('--dataset', default='drivAerML')
parser.add_argument('--eval', default=0, type=int,
                    help='1: standard evaluation, 2: decoupled inference (physical state caching + full mesh decoding)')
parser.add_argument('--max_ref_chunks', default=None, type=int,
                    help='Max chunks per run used during physical state caching (default: all)')
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--out-dim', default=4, type=int)
args = parser.parse_args()
print(args)

seed = 2
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# Training uses DDP; evaluation runs on a single GPU.
if not args.eval:
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
else:
    local_rank = 0
    device = torch.device("cuda", args.gpu)
    torch.cuda.set_device(args.gpu)

hparams = {'lr': args.lr, 'batch_size': args.batch_size, 'nb_epochs': args.nb_epochs}

with open(args.json_file, "r") as f:
    data_list = json.load(f)

val_dir = args.val_dir if args.val_dir is not None else args.data_dir

if not args.eval:
    train_dataset = DrivAerChunkDataset(root=args.data_dir, data_list=data_list, train=True)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DrivAerMLVTUChunkDataLoader(train_dataset, batch_size=args.batch_size,
                                               sampler=train_sampler, num_workers=32)

val_dataset = DrivAerChunkDataset(root=val_dir, data_list=data_list, train=False)
if args.eval:
    val_loader = DrivAerMLVTUChunkDataLoader(val_dataset, batch_size=args.batch_size,
                                             sampler=SequentialSampler(val_dataset),
                                             num_workers=8)
else:
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_loader = DrivAerMLVTUChunkDataLoader(val_dataset, batch_size=args.batch_size,
                                             sampler=val_sampler, num_workers=32)

pos_mean = None
pos_std = None
norm_mean = None
norm_std = None
out_mean = None
out_std = None

path = args.save_dir
if not os.path.exists(path):
    os.makedirs(path)

log_file = 'test.log' if args.eval else 'train.log'
logging.basicConfig(filename=os.path.join(path, log_file), level=logging.INFO,
                    filemode='w', format='%(asctime)s - %(message)s')
logging.info(args)


def _load_state_dict(path, device):
    checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, torch.nn.Module):
        checkpoint = checkpoint.module if hasattr(checkpoint, 'module') else checkpoint
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)}")

    # Strip 'module.' prefix added by DistributedDataParallel.
    if any(k.startswith('module.') for k in state_dict):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    return state_dict


if not args.eval:
    model = Transolver_chunk_opt_matrix_mul.Model(**MODEL_KWARGS).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model)
    logging.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Validation batches: {len(val_loader)}")

    model = train_surface.main(
        device, train_loader, val_loader, model, hparams, path,
        val_iter=args.val_iter, reg=args.weight,
        pos_norm=args.pos_norm, out_norm=args.out_norm, norm_norm=args.norm_norm,
        pos_mean=pos_mean, pos_std=pos_std, out_mean=out_mean, out_std=out_std,
        norm_mean=norm_mean, norm_std=norm_std, full=True, local_rank=local_rank)

elif args.eval == 1:
    # Standard evaluation (single GPU)
    model = Transolver_chunk_opt_matrix_mul.Model(**MODEL_KWARGS).to(device)
    model.load_state_dict(_load_state_dict(args.model_ckpt, device))
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Validation batches: {len(val_loader)}")

    loss_mse, loss_mae, loss_l2re = train_surface.test(
        device, model, val_loader,
        pos_norm=args.pos_norm, out_norm=args.out_norm, norm_norm=args.norm_norm,
        pos_mean=pos_mean, pos_std=pos_std, out_mean=out_mean, out_std=out_std,
        norm_mean=norm_mean, norm_std=norm_std, full=True)
    logging.info(f"Test MSE: {loss_mse}, MAE: {loss_mae}, L2RE: {loss_l2re}")
    print(f"Test MSE: {loss_mse}, MAE: {loss_mae}, L2RE: {loss_l2re}")

elif args.eval == 2:
    # Decoupled inference: physical state caching (Stage 1) + full mesh decoding (Stage 2)
    state_dict = _load_state_dict(args.model_ckpt, device)

    caching_model = Transolver_chunk_opt_matrix_mul_amortize.PhysicalStateCachingModel(**MODEL_KWARGS).to(device)
    caching_model.load_state_dict(state_dict)

    decoding_model = Transolver_chunk_opt_matrix_mul_amortize.FullMeshDecodingModel(**MODEL_KWARGS).to(device)
    decoding_model.load_state_dict(state_dict)

    print(f"Validation batches: {len(val_loader)}")
    loss_mse, loss_mae, loss_l2re = train_surface.test_decoupled_inference(
        device, caching_model, decoding_model,
        test_loader1=val_loader, test_loader2=val_loader,
        max_ref_chunks=args.max_ref_chunks)
    logging.info(f"Decoupled Inference Test MSE: {loss_mse}, MAE: {loss_mae}, L2RE: {loss_l2re}")
    print(f"Decoupled Inference Test MSE: {loss_mse}, MAE: {loss_mae}, L2RE: {loss_l2re}")
