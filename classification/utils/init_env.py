import os
import torch


def init_env(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = rank % torch.cuda.device_count()
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        world_size = args.world_size
        rank = args.rank
        args.gpu = args.rank % torch.cuda.device_count()
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(args.rank % num_gpus)

        os.environ['RANK'] = str(args.rank)
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        os.environ['WORLD_SIZE'] = str(args.world_size)

        import subprocess
        node_list = os.environ['SLURM_STEP_NODELIST']
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        # specify master port
        if hasattr(args, 'port'):
            os.environ['MASTER_PORT'] = str(args.port)
        elif 'MASTER_PORT' in os.environ:
            pass  # use MASTER_PORT in the environment variable
        else:
            # 29500 is torch.distributed default port
            os.environ['MASTER_PORT'] = '28506'
        # use MASTER_ADDR in the environment variable if it already exists
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = addr
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(args.gpu)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()