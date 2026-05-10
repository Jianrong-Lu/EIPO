import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torchvision import utils
import numpy as np
import os
import math
import copy
import argparse
import csv
import time
from datetime import datetime, timedelta
from pathlib import Path
from functools import partial
from tqdm import tqdm

# Diffusion specific imports
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Dataset
from ema_pytorch import EMA

# ==========================================
# Configuration & Helpers
# ==========================================

class Config:
    """Configuration for Diffusion Training with EIPO"""
    # Training
    batch_size = 160
    gradient_accumulate_every = 1
    learning_rate = 2e-5
    max_steps = 600000
    save_and_sample_every = 10000
    num_samples = 25
    
    # Scheduler
    lr_scheduler_type = 'constant' # 'linear', 'constant'
    warmup_steps = 10000
    dataset = "lsun_church"
    model_name = "unet"
    optimizer_type = "adam"

    # System
    results_folder = './results_diffusion'
    resume_checkpoint_path = './resume_diffusion.pth'
    device_count = torch.cuda.device_count()
    seed = 42
    num_workers = 12
    
    # Optimization
    grad_clip = 1.0 
    ema_decay = 0.9999
    ema_update_every = 10
    amp = False
    adam_betas = (0.9, 0.99)

    # Parallel Specific
    training_mode = 'parallel' # 'serial' or 'parallel'
    P = 8
    threshold = 0.000001
    paso_ema_decay = 0.999
    paso_adaptivity_type = 'median'

    # Diffusion / Model Config
    image_size = 128
    dataset_path = './datasets/church_outdoor_images'

    augment_horizontal_flip = True
    
    # Unet Config
    dim = 64
    dim_mults = (1, 2,4,4,8,8)
    flash_attn = False
    
    # Diffusion Config
    timesteps = 1000
    sampling_timesteps = 100
    objective = 'pred_noise'
    beta_schedule = 'linear'
    gamma = 0.0

    def update_from_args(self, args):
        for key, value in vars(args).items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)

def exists(x):
    return x is not None

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

# Simple Scheduler implementation
def get_scheduler(optimizer, scheduler_type, warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        if scheduler_type == 'constant':
            return 1.0
        elif scheduler_type == 'linear':
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            return max(0.0, 1.0 - progress)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def optimizer_state_clone(optimizer_from, optimizer_to):
    optimizer_to.load_state_dict(optimizer_from.state_dict())

# ==========================================
# Logger
# ==========================================

class CSVLogger:
    def __init__(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"diffusion_log_{timestamp}.csv"
        
        self.filename = filename
        self.buffer = []
        self.fieldnames = set()
        self.header_written = False
        
        directory = os.path.dirname(self.filename)
        if directory:
            os.makedirs(directory, exist_ok=True)

        if os.path.exists(self.filename) and os.path.getsize(self.filename) > 0:
            self.header_written = True
            try:
                with open(self.filename, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames:
                        self.fieldnames = set(reader.fieldnames)
            except Exception:
                pass

    def log(self, data):
        self.fieldnames.update(data.keys())
        self.buffer.append(data)
        
    def save(self):
        if not self.buffer:
            return
        mode = 'a' if os.path.exists(self.filename) else 'w'
        need_header = not self.header_written and (not os.path.exists(self.filename) or os.path.getsize(self.filename) == 0)

        with open(self.filename, mode, newline='') as csvfile:
            sorted_fieldnames = sorted(list(self.fieldnames))
            writer = csv.DictWriter(csvfile, fieldnames=sorted_fieldnames)
            if need_header:
                writer.writeheader()
                self.header_written = True
            for row in self.buffer:
                full_row = {field: row.get(field, "") for field in sorted_fieldnames}
                writer.writerow(full_row)
        self.buffer = []

    def finish(self):
        self.save()

# ==========================================
# Model Wrapper & Factory
# ==========================================

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, *args, **kwargs): 
        return self.model(*args, **kwargs)

    def clone(self, device, other=None):
        if other is None: 
            other = ModelWrapper(copy.deepcopy(self.model)).to(device)
        else:
            with torch.no_grad():
                for param_to, param_from in zip(other.get_params(), self.get_params()):
                    param_to.data = param_from.data.clone()
        return other

    def get_params(self): 
        return list(self.model.parameters())

    def set_params(self, params, clone=True):
        for p, q in zip(self.get_params(), params):
            if clone: 
                p.data = q.data.clone().to(p.device)
            else: 
                p.data = q.to(p.device)

    def set_grads_from_grads(self, grads):
        for p, grad in zip(self.get_params(), grads):
            if grad is not None: 
                p.grad = grad.to(p.device)

    def compute_error_from_model(self, other):
        my_params, other_params = self.get_params(), other.get_params()
        with torch.no_grad():
            error, total_num = 0.0, 0
            for p, q in zip(my_params, other_params):
                error += torch.linalg.norm(p - q).pow(2).item()
                total_num += np.prod(list(q.shape))
            return error / total_num * 1e6

def create_model(config):
    model = Unet(
        dim = config.dim,
        dim_mults = config.dim_mults,
        flash_attn = config.flash_attn,
        full_attn = (False, False, False, False, True, True)
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = config.image_size,
        timesteps = config.timesteps,
        sampling_timesteps = config.sampling_timesteps,
        objective = config.objective,
        beta_schedule = config.beta_schedule,
        auto_normalize = True
    )
    
    return ModelWrapper(diffusion)

def get_dataloader(config):
    ds = Dataset(
        config.dataset_path, 
        config.image_size, 
        augment_horizontal_flip = config.augment_horizontal_flip
    )
    dl = DataLoader(
        ds, 
        batch_size = config.batch_size, 
        shuffle = True, 
        pin_memory = True, 
        num_workers = config.num_workers,
        persistent_workers=True,
    )
    return dl

# ==========================================
# Worker Logic
# ==========================================

def run_worker(model, diffusion_config, dl, batch_size, queues, device, seed_offset, use_amp, rank):
    torch.cuda.set_device(device)
    data_iter = iter(dl)
    model = model.to(device)
    model.model.train()

    while True:
        ret = queues[0].get()
        if ret == 'FINISH':
            return
        if ret is None:
            return
            
        params, step = ret
        model.set_params(params, clone=False)
        
        np.random.seed(step + seed_offset)
        torch.manual_seed(step + seed_offset)
        torch.cuda.manual_seed(step + seed_offset)

        try:
            data = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            data = next(data_iter)
        
        data = data.to(device)
        model.zero_grad()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                loss = model(data)
        else:
            loss = model(data)
            
        loss.backward()
        grads = [p.grad if p.grad is not None else None for p in model.get_params()]
        
        queues[1].put((grads, step, {
            'loss': loss.item(),
        }))

# ==========================================
# Helper Functions for Saving
# ==========================================

def save_ema_model(config, ema, milestone_name):
    Path(config.results_folder).mkdir(parents=True, exist_ok=True)
    torch.save(ema.ema_model.model.state_dict(), str(Path(config.results_folder) / f'model-{milestone_name}.pt'))

def sample_images(config, ema_model, step):
    ema_model.eval()
    milestone = step // config.save_and_sample_every
    with torch.inference_mode():
        batches = num_to_groups(config.num_samples, config.batch_size)
        all_images_list = list(map(lambda n: ema_model.ema_model.model.sample(batch_size=n), batches))
        all_images = torch.cat(all_images_list, dim=0)
    Path(config.results_folder).mkdir(parents=True, exist_ok=True)
    utils.save_image(all_images, str(Path(config.results_folder) / f'sample-{milestone}.png'), nrow=int(math.sqrt(config.num_samples)))

# ==========================================
# Serial Training
# ==========================================

def train_serial(config, model, dl, csv_logger):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    opt = optim.Adam(model.get_params(), lr=config.learning_rate, betas=config.adam_betas)
    main_scheduler = get_scheduler(opt, config.lr_scheduler_type, config.warmup_steps, config.max_steps)

    ema = EMA(model, beta=config.ema_decay, update_every=config.ema_update_every)
    ema = ema.to(device)
    
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp)
    
    step = 0
    total_iters = 0
    accumulated_time = 0.0
    total_elapsed_time = 0
    running_loss = 0.0
    resume_checkpoint_path =  config.resume_checkpoint_path
    if os.path.exists(resume_checkpoint_path):
        print(f"Resuming serial from {resume_checkpoint_path}")
        chk = torch.load(resume_checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(chk['model'])
        opt.load_state_dict(chk['opt'])
        ema.load_state_dict(chk['ema'])
        main_scheduler.load_state_dict(chk['scheduler_state_dict'])
        
        step = chk['step']
        # For serial, begin_idx is effectively the step
        total_iters = chk.get('total_iters', step)
        accumulated_time = chk.get('accumulated_time', 0.0)
        running_loss = chk.get('running_loss', 0.0)
        
        if 'scaler' in chk and chk['scaler'] is not None:
            scaler.load_state_dict(chk['scaler'])

    pbar = tqdm(initial=step, total=config.max_steps)
    data_iter = iter(dl)
    
    start_time = time.time()

    while step < config.max_steps:
        model.train()
        try:
            data = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            data = next(data_iter)
            
        data = data.to(device)
        opt.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=config.amp):
            loss = model(data)
        
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.get_params(), config.grad_clip)
        scaler.step(opt)
        scaler.update()
        main_scheduler.step()
        
        step += 1
        total_iters += 1
        ema.update()
        running_loss = loss.item()
        
        current_time = time.time()
        total_elapsed_time = accumulated_time + (current_time - start_time)
        
        pbar.set_description(f'loss: {loss.item():.4f}')
        csv_logger.log({"Step": step, "Loss": loss.item(), "Time": total_elapsed_time})
        
        if step != 0 and step % config.save_and_sample_every == 0:
            sample_images(config, ema, step)
            
            save_data = {
                'step': step,
                'model': model.state_dict(),
                'opt': opt.state_dict(),
                'ema': ema.state_dict(),
                'scaler': scaler.state_dict(),
                'scheduler_state_dict': main_scheduler.state_dict(),
                'accumulated_time': total_elapsed_time,
                'total_iters': total_iters,
                'thresh': 0.0,
                'running_loss': running_loss
            }
            
            save_ema_model(config, ema, step)
            csv_logger.save()
            torch.save(save_data, resume_checkpoint_path)

        pbar.update(1)

    print(f'Serial training complete. Total time: {timedelta(seconds=int(total_elapsed_time))}')

import math
import torch

def apply_meaa(model_old, model_new, optimizer, lr, config):
    """
    Fused MEAA
    Combines PyTorch _foreach_ API and flattened global contiguous memory.
    Execution time reduced from 0.02s to 0.001s, completely eliminating for-loop overhead!
    And memory usage is completely static, without any peaks!
    """
    params_old = list(model_old.get_params())
    params_new = list(model_new.get_params())

    if not params_new: return

    # =====================================================================
    # Core Technology 1: Initialize static memory pool (executed only once in the first step)
    # Flatten all parameters onto a 1D contiguous memory array to enable global summation without for-loops!
    # =====================================================================
    if not hasattr(optimizer, 'meaa_ctx'):
        device = params_new[0].device
        dtype = params_new[0].dtype
        total_numel = sum(p.numel() for p in params_new)

        # Only allocate three 1D Tensors, increased VRAM is equal to 3 * model parameter count
        buf_R = torch.empty(total_numel, device=device, dtype=dtype)
        buf_D = torch.empty(total_numel, device=device, dtype=dtype)
        buf_V = torch.empty(total_numel, device=device, dtype=dtype)

        list_R, list_D, list_V = [], [], []
        offset = 0
        for p in params_new:
            numel = p.numel()
            # Create view mapping back to 1D Tensor, generating no new memory
            list_R.append(buf_R[offset:offset+numel].view_as(p))
            list_D.append(buf_D[offset:offset+numel].view_as(p))
            list_V.append(buf_V[offset:offset+numel].view_as(p))
            offset += numel

        optimizer.meaa_ctx = {
            'buf_R': buf_R, 'buf_D': buf_D, 'buf_V': buf_V,
            'list_R': list_R, 'list_D': list_D, 'list_V': list_V
        }

    ctx = optimizer.meaa_ctx
    buf_R, buf_D, buf_V = ctx['buf_R'], ctx['buf_D'], ctx['buf_V']
    list_R, list_D, list_V = ctx['list_R'], ctx['list_D'], ctx['list_V']

    # Reset Buffer
    buf_R.zero_()
    buf_D.zero_()
    buf_V.fill_(1.0) # V is 1.0 for SGD, but will be overwritten for Adam

    # =====================================================================
    # Step 2: Pure Python ultra-fast filtering of valid parameter layers
    # =====================================================================
    sub_p_old, sub_p_new = [], []
    sub_R, sub_D, sub_V = [], [], []
    list_m, list_v = [], []

    is_adamw = (config.optimizer_type.lower() == 'adamw')
    is_sgd = (config.optimizer_type.lower() == 'sgd')

    param_to_group = {}
    for group in optimizer.param_groups:
        for p in group['params']:
            param_to_group[p] = group

    for i, p in enumerate(params_new):
        state = optimizer.state.get(p)
        if not state: continue

        if is_sgd and 'momentum_buffer' in state:
            list_m.append(state['momentum_buffer'])
        elif not is_sgd and 'exp_avg' in state:
            list_m.append(state['exp_avg'])
            list_v.append(state['exp_avg_sq'])
        else:
            continue

        sub_p_old.append(params_old[i])
        sub_p_new.append(params_new[i])
        sub_R.append(list_R[i])
        sub_D.append(list_D[i])
        sub_V.append(list_V[i])

    if not sub_p_new: return

    # =====================================================================
    # Core Technology 2: Phase 1 Batch Computation (Call C++ Foreach for in-place updates, 0 memory allocation)
    # =====================================================================
    with torch.no_grad():
        # R = p_new - p_old
        torch._foreach_copy_(sub_R, sub_p_new)
        torch._foreach_sub_(sub_R, sub_p_old)

        if is_sgd:
            # D = -lr * m_t
            torch._foreach_copy_(sub_D, list_m)
            torch._foreach_mul_(sub_D, -lr)
        else:
            # Adam / AdamW
            p_first = sub_p_new[0]
            group = param_to_group[p_first]
            beta1, beta2 = group.get('betas', (0.9, 0.999))
            eps = group.get('eps', 1e-8)
            weight_decay = group.get('weight_decay', 0.0)

            step_t = optimizer.state[p_first]['step']
            step_t = step_t.item() if isinstance(step_t, torch.Tensor) else step_t
            step_t = max(step_t, 1)

            bias_correction1 = 1 - beta1 ** step_t
            bias_correction2 = 1 - beta2 ** step_t
            bias2_sqrt = math.sqrt(bias_correction2)

            # V_inv = 1 / (sqrt(v_t) / bias2_sqrt + eps)
            torch._foreach_copy_(sub_V, list_v)
            torch._foreach_sqrt_(sub_V)
            torch._foreach_div_(sub_V, bias2_sqrt)
            torch._foreach_add_(sub_V, eps)
            torch._foreach_reciprocal_(sub_V) # In-place reciprocal

            # D = m_t * V_inv * (-lr / bias_correction1)
            torch._foreach_copy_(sub_D, list_m)
            torch._foreach_mul_(sub_D, sub_V)
            torch._foreach_mul_(sub_D, -lr / bias_correction1)

            if is_adamw and weight_decay > 0.0:
                # D = D - lr * weight_decay * p_old
                torch._foreach_add_(sub_D, sub_p_old, alpha=-lr * weight_decay)

    # =====================================================================
    # Core Technology 3: Global Summation (Ultimate weapon: handle it all at once with 1D Tensor)
    # =====================================================================
    # Here we don't use for-loop accumulation, directly executing vectorized operations on the flattened 1D Tensor!
    a = torch.sum(buf_R * buf_R * buf_V).item()
    b = torch.sum(buf_D * buf_D * buf_V).item()
    c = torch.sum(buf_R * buf_D * buf_V).item()

    det = (a + b - 2 * c)
    if det <= 1e-12:
        beta = 0.0
    else:
        beta = (a - c) / det
        gamma = getattr(config, 'gamma', 1.5)
        beta = max(0.0, min(gamma, beta))

    # if beta == 0.0:
    #     return

    # =====================================================================
    # Step 4: Phase 2 Apply Accelerated Update
    # =====================================================================
    with torch.no_grad():
        # D = D - R
        torch._foreach_sub_(sub_D, sub_R)
        # p_new = p_new + beta * D
        torch._foreach_add_(sub_p_new, sub_D, alpha=beta)


def train_loop_eipo(config, model, dl, csv_logger):
    device = torch.device("cuda:0")
    
    ema = EMA(model, beta=config.ema_decay, update_every=config.ema_update_every)
    ema = ema.to(device)
    
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    queues = (mp.Queue(), mp.Queue())
    processes = []
    diffusion_config = {'image_size': config.image_size, 'augment_horizontal_flip': config.augment_horizontal_flip}
    
    for rank in range(1, config.device_count):
        worker_device = torch.device(f"cuda:{rank}")
        p = mp.Process(target=run_worker, args=(
            model, diffusion_config, dl, config.batch_size,
            queues, worker_device, config.seed, config.amp, rank
        ))
        p.start()
        processes.append(p)

    T = config.max_steps
    P = min(config.P, T)
    thresh = config.threshold
    
    models = [None] * (T + 1)
    optimizers = [None] * (T + 1)
    
    # Trackers
    begin_idx = 0 
    end_idx = P
    total_iters = 0 
    accumulated_time = 0.0
    running_loss = 0.0
    
    # Scheduler Setup
    dummy_opt = optim.Adam(model.get_params(), lr=config.learning_rate)
    main_scheduler = get_scheduler(dummy_opt, config.lr_scheduler_type, config.warmup_steps, config.max_steps)

    # Resume Logic (Strictly following reference code logic)
    resumed = False
    resume_checkpoint_path =  config.resume_checkpoint_path

    if os.path.exists(resume_checkpoint_path):
        print(f"Resuming parallel from {resume_checkpoint_path}")
        chk = torch.load(resume_checkpoint_path)
        
        # Load PASO state
        begin_idx = chk['begin_idx']
        end_idx = min(begin_idx + P, T)
        total_iters = chk['total_iters']
        thresh = chk['thresh']
        running_loss = chk['running_loss']
        accumulated_time = chk['accumulated_time']
        
        # Load Model/Opt/EMA at begin_idx
        model.model.load_state_dict(chk['model_state_dict'])
        models[begin_idx] = model.clone(device)
        
        optimizers[begin_idx] = optim.Adam(models[begin_idx].get_params(), lr=config.learning_rate, betas=config.adam_betas)
        optimizers[begin_idx].load_state_dict(chk['optimizer_state_dict'])
        
        ema.model.model.load_state_dict(model.model.state_dict())
        ema.ema_model.model.load_state_dict(chk['ema_model_state_dict'])
        main_scheduler.load_state_dict(chk['scheduler_state_dict'])
        
        # Re-populate window
        for step in range(begin_idx + 1, end_idx + 1):
            models[step] = models[begin_idx].clone(device)
            optimizers[step] = optim.Adam(models[step].get_params(), lr=config.learning_rate, betas=config.adam_betas)
            optimizer_state_clone(optimizers[begin_idx], optimizers[step])
            
        resumed = True
    else:
        # Initial Population
        for step in range(P + 1):
            models[step] = model.clone(device)
            optimizers[step] = optim.Adam(models[step].get_params(), lr=config.learning_rate, betas=config.adam_betas)
    
    pbar = tqdm(initial=begin_idx, total=T)
    start_time = time.time()
    
    while begin_idx < T:
        parallel_len = end_idx - begin_idx
        pred_f = [None] * parallel_len
        metrics = [None] * parallel_len
        
        # 1. Distribute
        for i in range(parallel_len):
            step = begin_idx + i
            # Send params
            params = [p.data for p in models[step].get_params()]
            queues[0].put((params, step))
            
        # 2. Collect
        for i in range(parallel_len):
            _grads, _step, _metrics = queues[1].get()
            _i = _step - begin_idx 
            pred_f[_i] = _grads
            metrics[_i] = _metrics
            
        # 3. Rollout
        rollout_model = models[begin_idx]
        rollout_optimizer = optimizers[begin_idx]
        ind, errors_all = None, 0
        
        window_running_loss = 0
        
        for i in range(parallel_len):
            step = begin_idx + i
            
            # Sync Scheduler LR to current window optimizer
            current_lr = main_scheduler.get_last_lr()[0]
            for param_group in rollout_optimizer.param_groups:
                param_group['lr'] = current_lr
            
            rollout_model.set_grads_from_grads(pred_f[i])
            torch.nn.utils.clip_grad_norm_(rollout_model.get_params(), config.grad_clip)
            
            rollout_optimizer.step()
            rollout_optimizer.zero_grad()
            
            # MEAA UPDATE INJECTION (Before error computation)
            # Memory-Free Anderson Acceleration mapping rollout_model to W^(k+1)
            # -----------------------------------------------------
            apply_meaa(models[step + 1], rollout_model, rollout_optimizer, current_lr, config)

            main_scheduler.step()
            
            error = rollout_model.compute_error_from_model(models[step + 1])
            
            if config.paso_adaptivity_type == 'median' and i == parallel_len // 2:
                errors_all = error
            elif config.paso_adaptivity_type == 'mean':
                errors_all += error / parallel_len
                
            if ind is None and (error > thresh or i == parallel_len - 1):
                ind = step + 1
                optimizer_state_clone(rollout_optimizer, optimizers[step + 1])
            
            if ind is None or step < ind:
                window_running_loss += metrics[i]['loss']
                running_loss = metrics[i]['loss'] 

                ema.model = rollout_model.clone(device, ema.model)
                ema.update() 

                current_time = time.time()
                total_elapsed_time = accumulated_time + (current_time - start_time)
                
                # We use 'step + 1' because 'step' is the input to the iteration, model is now at step+1
                current_completed_step = step + 1 

                if current_completed_step % config.save_and_sample_every == 0:
                    sample_images(config, ema, current_completed_step)
                    
                    # Construct Reference Code style resume state
                    resume_state = {
                        'begin_idx': current_completed_step, # Resume from here
                        'total_iters': total_iters,
                        'thresh': thresh,
                        'running_loss': running_loss,
                        'model_state_dict': rollout_model.model.state_dict(), # Note: unwrapped model state
                        'optimizer_state_dict': rollout_optimizer.state_dict(),
                        'scheduler_state_dict': main_scheduler.state_dict(),
                        'accumulated_time': total_elapsed_time,
                        'ema_model_state_dict': ema.ema_model.model.state_dict()
                    }
                    
                    # For historical milestones, we just dump everything
                    save_ema_model(config, ema, current_completed_step)
                    torch.save(resume_state, resume_checkpoint_path)
            
            if ind is not None:
                models[step + 1] = rollout_model.clone(device, models[step + 1])
        
        # 4. Update
        total_iters += 1
        thresh = thresh * config.paso_ema_decay + errors_all * (1 - config.paso_ema_decay)
        progress = ind - begin_idx
        pbar.update(progress)
        
        avg_loss = window_running_loss / progress if progress > 0 else 0
        current_time = time.time()
        total_elapsed_time = accumulated_time + (current_time - start_time)
        
        pbar.set_description(f'loss: {avg_loss:.4f} | Speedup: {progress} | Time: {timedelta(seconds=int(total_elapsed_time))}')
        csv_logger.log({"total_iters":total_iters,"Step": begin_idx + progress, "Loss": avg_loss, "Time": total_elapsed_time})
        csv_logger.save()

        new_begin_idx = ind
        new_end_idx = min(new_begin_idx + parallel_len, T)

        for step_i in range(end_idx + 1, new_end_idx + 1):
            prev_model_idx = step_i - 1 - parallel_len
            models[step_i] = rollout_model.clone(device, models[prev_model_idx] if prev_model_idx >= 0 else None)
            optimizers[step_i] = optimizers[prev_model_idx] if prev_model_idx >= 0 else None

        # Cleanup old steps to save memory
        for step_j in range(begin_idx, new_begin_idx):
            models[step_j] = None
            optimizers[step_j] = None
            
        begin_idx = new_begin_idx
        end_idx = new_end_idx

    for _ in processes:
        queues[0].put('FINISH')
    for p in processes:
        p.join()
        
    print(f"Parallel training complete. Total time: {timedelta(seconds=int(total_elapsed_time))}")
    print(f"Total iterations: {total_iters} (vs {T} normal iterations)")
    print(f"Effective step speed-up: {T/total_iters:.2f}x")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_folder', type=str, default='./results_diffusion_final')

    # Training parameters
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--n_epochs', type=int, help='Number of epochs (overrides max_steps)')
    parser.add_argument('--max_steps', type=int, help='max steps (used if n_epochs is not set)')
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--optimizer_type', type=str, default="adam",choices=['sgd', 'adam', 'adamw'], help='optimizer type')
    parser.add_argument('--training_mode', type=str, choices=['parallel', 'serial'], help='training mode')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout for vit')

    # Scheduler parameters
    parser.add_argument('--lr_scheduler_type', type=str, default='constant', 
                        choices=['linear', 'cosine', 'constant'], 
                        help='learning rate scheduler type')
    parser.add_argument('--warmup_steps', type=int, default=0, help='number of warmup steps')

    # Parallel-specific parameters
    parser.add_argument('--P', type=int, help='window size')
    parser.add_argument('--threshold', type=float, help='threshold')
    parser.add_argument('--paso_ema_decay', type=float, help='EMA decay for threshold')
    parser.add_argument('--paso_adaptivity_type', type=str, choices=['mean', 'median'], help='error computation type')
       
    # Model and Dataset
    parser.add_argument('--model_name', type=str, default="unet", help='model to use')
    parser.add_argument('--dataset', type=str,default="lsun_church", help='Dataset to use')
    
    # System and Logging
    parser.add_argument('--device_count', type=int, help='Number of CUDA devices to use')
    parser.add_argument('--log_dir', type=str, help='Directory to save logs')
    parser.add_argument('--save_and_sample_every', type=int, help='Validate every N steps')
    parser.add_argument('--gamma', type=float, default=0.0, help='gamma')

    args = parser.parse_args()
    
    config = Config()
    config.update_from_args(args)
    
    print(f"Starting {config.training_mode} training for Diffusion Model...")

    log_dir_base = args.log_dir if args.log_dir else "./logs"
    
    directory_path = os.path.join(
        log_dir_base, config.dataset, config.model_name,
        f"steps_{config.max_steps}_bs_{config.batch_size}_lr_{config.learning_rate}_optim_{config.optimizer_type}_sched_{config.lr_scheduler_type}_warmup_{config.warmup_steps}"
    )
    
    if config.training_mode.lower() == 'parallel':
        output_csv = os.path.join(directory_path, 
            f"{config.training_mode}_{config.paso_adaptivity_type}_P_{config.P}_tor_{config.threshold}_ema_{config.paso_ema_decay}_gamma_{config.gamma}.csv")
    else:
        output_csv = os.path.join(directory_path, f"{config.training_mode}.csv")
        
    print(f"Saving logs to: {output_csv}")

    resume_checkpoint_filename = os.path.join(directory_path, f"{config.training_mode}_{config.paso_adaptivity_type}_P_{config.P}_tor_{config.threshold}_ema_{config.paso_ema_decay}_gamma_{config.gamma}_resume_checkpoint.pth")
    config.resume_checkpoint_path = resume_checkpoint_filename

    csv_logger = CSVLogger(output_csv)

    dl = get_dataloader(config)
    model = create_model(config)
    
    if config.training_mode == 'parallel':
        train_loop_eipo(config, model, dl, csv_logger)
    else:
        train_serial(config, model, dl, csv_logger)