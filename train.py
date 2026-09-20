"""Progressive training of ReDNet.

Example (single GPU):
    python train.py -opt options/train_rednet.yml
Example (2 GPUs):
    torchrun --nproc_per_node=2 --master_port=4321 train.py -opt options/train_rednet.yml --launcher pytorch
"""
import argparse
import logging
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from rednet.archs.rednet_arch import ReDNet
from rednet.dataset import DirtTrainDataset, PairedValDataset
from rednet.losses import DirtLoss
from rednet.utils import pad_to_multiple, psnr


def get_position_from_periods(iteration, cumulative_period):
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i
    return len(cumulative_period) - 1


class CosineAnnealingRestartCyclicLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, periods, restart_weights=(1, ), eta_mins=(0, ), last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_mins = eta_mins
        assert len(self.periods) == len(self.restart_weights)
        self.cumulative_period = [sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))]
        super(CosineAnnealingRestartCyclicLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch, self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]
        eta_min = self.eta_mins[idx]
        return [
            eta_min + current_weight * 0.5 * (base_lr - eta_min) *
            (1 + math.cos(math.pi * ((self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to the YAML option file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none')
    args = parser.parse_args()
    with open(args.opt, 'r') as f:
        opt = yaml.safe_load(f)
    opt['is_train'] = True
    opt['launcher'] = args.launcher
    if 'dist' not in opt:
        opt['dist'] = args.launcher != 'none'
    return opt


def init_dist():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
        return rank, world_size, True
    return 0, 1, False


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    opt = parse_options()
    rank, world_size, dist_enabled = init_dist()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    is_main = (rank == 0)

    set_random_seed(opt.get('manual_seed', 100))

    exp_name = opt['name']
    exp_dir = os.path.join(opt.get('exp_root', 'experiments'), exp_name)
    ckpt_dir = os.path.join(exp_dir, 'models')
    state_dir = os.path.join(exp_dir, 'training_states')
    vis_dir = os.path.join(exp_dir, 'visualization')
    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(state_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        log_file = os.path.join(exp_dir, f'train_{exp_name}_{time.strftime("%Y%m%d_%H%M%S")}.log')
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)s: %(message)s',
                            handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
        logging.info(f'Options:\n{yaml.safe_dump(opt)}')

    # ---------------- model ----------------
    net_opt = dict(opt['network_g'])
    net_opt.pop('type', None)
    model = ReDNet(**net_opt).to(device)
    if dist_enabled:
        model = DDP(model, device_ids=[torch.cuda.current_device()])

    # ---------------- loss / optimizer ----------------
    train_opt = opt['train']
    pixel_opt = dict(train_opt.get('pixel_opt', {}))
    pixel_opt.pop('type', None)
    pixel_opt.pop('reduction', None)
    cri_pix = DirtLoss(**pixel_opt).to(device)

    optim_opt = train_opt['optim_g']
    optimizer = torch.optim.AdamW(model.parameters(), lr=optim_opt['lr'],
                                  weight_decay=optim_opt.get('weight_decay', 1e-4),
                                  betas=tuple(optim_opt.get('betas', [0.9, 0.999])))
    sch_opt = train_opt['scheduler']
    scheduler = CosineAnnealingRestartCyclicLR(optimizer,
                                               periods=sch_opt['periods'],
                                               restart_weights=sch_opt['restart_weights'],
                                               eta_mins=sch_opt['eta_mins'])

    # ---------------- resume ----------------
    current_iter = 0
    resume_state = opt.get('path', {}).get('resume_state')
    if resume_state and os.path.isfile(resume_state):
        state = torch.load(resume_state, map_location='cpu')
        model_module = model.module if dist_enabled else model
        model_module.load_state_dict(state['params'])
        optimizer.load_state_dict(state['optimizers'])
        scheduler.load_state_dict(state['schedulers'])
        current_iter = state['iter']
        if is_main:
            logging.info(f'Resumed from {resume_state} (iter {current_iter})')

    # ---------------- data ----------------
    ds_opt = opt['datasets']['train']
    train_set = DirtTrainDataset(dataroot_gt=ds_opt['dataroot_gt'],
                                 dataroot_lq=ds_opt['dataroot_lq'],
                                 dataroot_mask=ds_opt.get('dataroot_mask'),
                                 gt_size=ds_opt.get('gt_size', 384),
                                 geometric_augs=ds_opt.get('geometric_augs', True),
                                 use_mask=ds_opt.get('use_mask', True))
    if dist_enabled:
        sampler = DistributedSampler(train_set, shuffle=True)
    else:
        sampler = None
    train_loader = DataLoader(train_set,
                              batch_size=ds_opt.get('batch_size_per_gpu', 2),
                              shuffle=(sampler is None),
                              sampler=sampler,
                              num_workers=ds_opt.get('num_worker_per_gpu', 2),
                              drop_last=True,
                              pin_memory=True)

    val_loader = None
    if opt.get('datasets', {}).get('val'):
        val_opt = opt['datasets']['val']
        val_set = PairedValDataset(val_opt['dataroot_lq'], val_opt['dataroot_gt'])
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)

    # ---------------- progressive schedule ----------------
    iters = ds_opt['iters']                      # e.g. [64000, 48000, 36000, 36000, 24000]
    mini_batch_sizes = ds_opt['mini_batch_sizes']  # per GPU
    mini_gt_sizes = ds_opt['gt_sizes']
    base_batch = ds_opt.get('batch_size_per_gpu', 2)
    gt_size = ds_opt.get('gt_size', 384)
    groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])

    total_iters = int(train_opt['total_iter'])
    save_freq = int(opt['logger'].get('save_checkpoint_freq', 4000))
    val_freq = int(opt.get('val', {}).get('val_freq', 4000))
    print_freq = int(opt['logger'].get('print_freq', 1000))
    use_grad_clip = train_opt.get('use_grad_clip', True)

    def save_ckpt(iter_i, epoch_i=-1, latest=False):
        if not is_main:
            return
        model_module = model.module if dist_enabled else model
        tag = 'latest' if latest else str(iter_i)
        torch.save({'params': model_module.state_dict()},
                   os.path.join(ckpt_dir, f'net_g_{tag}.pth'))
        torch.save({'params': model_module.state_dict(),
                    'optimizers': optimizer.state_dict(),
                    'schedulers': scheduler.state_dict(),
                    'iter': iter_i, 'epoch': epoch_i},
                   os.path.join(state_dir, f'{iter_i}.state'))

    @torch.no_grad()
    def validate(iter_i):
        if val_loader is None or not is_main:
            return
        model_module = model.module if dist_enabled else model
        model_module.eval()
        values = []
        for batch in val_loader:
            lq = batch['lq'].to(device)
            gt = batch['gt'][0].permute(1, 2, 0).numpy()
            h, w = lq.shape[-2:]
            inp = pad_to_multiple(lq, factor=8)
            out, _ = model_module(inp)
            out = torch.clamp(out[:, :, :h, :w], 0, 1)
            out = out[0].permute(1, 2, 0).cpu().numpy()
            values.append(psnr(out, gt))
        model_module.train()
        logging.info(f'Validation ValSet, iter {iter_i}, # psnr: {np.mean(values):.4f}')

    logging.info(f'Start training: iters {total_iters}, world size {world_size}, '
                 f'progressive groups {list(groups)}')
    validate(current_iter) if current_iter > 0 else None

    model.train()
    epoch = 0
    last_validated = current_iter
    t_iter = time.time()
    while current_iter < total_iters:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in train_loader:
            current_iter += 1
            if current_iter > total_iters:
                break

            scheduler.step()

            # progressive learning: pick stage
            j = ((current_iter > groups) != True).nonzero()[0]
            bs_j = len(groups) - 1 if len(j) == 0 else j[0]
            mini_gt_size = mini_gt_sizes[bs_j]
            mini_batch_size = mini_batch_sizes[bs_j]

            lq, gt, mask = batch['lq'], batch['gt'], batch['mask']
            if mini_batch_size < base_batch:
                indices = random.sample(range(0, base_batch), k=mini_batch_size)
                lq = lq[indices]
                mask = mask[indices]
                gt = gt[indices]
            if mini_gt_size < gt_size:
                x0 = int((gt_size - mini_gt_size) * random.random())
                y0 = int((gt_size - mini_gt_size) * random.random())
                x1, y1 = x0 + mini_gt_size, y0 + mini_gt_size
                lq = lq[:, :, x0:x1, y0:y1]
                mask = mask[:, :, x0:x1, y0:y1]
                gt = gt[:, :, x0:x1, y0:y1]

            lq = lq.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            optimizer.zero_grad()
            output, output_mask = model(lq)
            loss = cri_pix(output, gt, output_mask, mask.to(device, non_blocking=True))
            loss.backward()
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            optimizer.step()

            if is_main and current_iter % print_freq == 0:
                lr_now = optimizer.param_groups[0]['lr']
                logging.info(f'[epoch: {epoch}, iter: {current_iter}, lr: ({lr_now:.3e})] '
                             f'l_pix: {loss.item():.4e} '
                             f'(stage {bs_j}: patch {mini_gt_size}, bs {mini_batch_size * world_size})')
            if is_main and current_iter % save_freq == 0:
                save_ckpt(current_iter, epoch)
                last_validated = current_iter
                validate(current_iter)
        epoch += 1

    if is_main:
        save_ckpt(total_iters, epoch, latest=True)
        if last_validated < total_iters:
            validate(total_iters)
        logging.info('End of training.')
        if dist_enabled:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
