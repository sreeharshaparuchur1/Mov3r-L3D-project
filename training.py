import torch
import torch.distributed as dist 
from tqdm import tqdm
import numpy as np
import tensorboardX
import os
import argparse
import time
from torch.utils.data.distributed import DistributedSampler
import os
import datetime
from torch.utils.data import Dataset, DataLoader
from cross_attention import PatchWiseCrossAttentionDecoder, DUSt3RAsymmetricCrossAttention

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="pointmap_predict", help="Run name, default: pointmap_predict")
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_patches', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device_ids', nargs='+', type=int, default=[0, 1, 2], help='List of GPU device IDs')
    parser.add_argument('--log_dir', type=str, default='/data/kmirakho/git/Mov3r-L3D-project/logs/')
    parser.add_argument('--model_dir', type=str, default='/data/kmirakho/git/Mov3r-L3D-project/models/')
    parser.add_argument('--load_model', action="store_true")
    parser.add_argument('--run_eval', action="store_true")
    parser.add_argument('--eval_model', type=str, default='/data/kmirakho/offline_data/model/data_10_40_640/model_20.pth')
    parser.add_argument('--dataset_path', type=str, default='/data/kmirakho/git/Mov3r-L3D-project/data/')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args

def create_distributed_loader(dataset, batch_size, local_rank, world_size):
    # dataset = #load the dataset here
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    ), sampler

if __name__ == '__main__':
    args = get_config()
    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set the device IDs for distributed training
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.device_ids))
    # Set the number of threads for PyTorch
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    # Set NCCL environment variables
    os.environ['NCCL_ALGO'] = 'Ring'
    os.environ['NCCL_NSOCKS_PERTHREAD'] = '4'
    os.environ['NCCL_SOCKET_NTHREADS'] = '2'
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])

    # Create distributed data loader
    # train_loader, sampler = create_distributed_loader(
    #     dataset=None,  # Replace with your dataset
    #     batch_size=args.batch_size,
    #     local_rank=local_rank,
    #     world_size=world_size
    # )

    # Initialize process group first
    dist.init_process_group(
        backend='nccl',  # Use 'gloo' for CPU
        init_method='env://',
        world_size=world_size,
        rank=local_rank,
        timeout=datetime.timedelta(seconds=30)
        )
    
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    log_dir = args.log_dir + args.run_name
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    writer = tensorboardX.SummaryWriter(log_dir)

    # Initialize ViT model for self attention

    # Initialize the CrossAttention model

    x1 = torch.randn(128, 256, 512) #batch size, (num patches x num_patches), embed dim
    x2 = torch.randn(128, 256, 512) #batch size, (num patches x num_patches), embed dim
    dust3r_cross = DUSt3RAsymmetricCrossAttention(
        encoder_dim=512, 
        decoder_dim=512, 
        depth=3, 
        num_heads=8, 
        dropout=0.0
    ).cuda(device)

    if torch.cuda.device_count() > 1:
        dust3r_cross = torch.nn.parallel.DistributedDataParallel(dust3r_cross, device_ids=[local_rank])
    x1 = x1.cuda(local_rank)
    x2 = x2.cuda(local_rank)
    dust3r_cross(x1, x2)
    # Initialize the PointMap & Depth Predictor model