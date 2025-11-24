import os
import time
import argparse
import math
from numpy import finfo

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import Tacotron2
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger
from hparams import create_hparams


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")

def prepare_dataloaders(hparams):
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset   = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset, shuffle=True)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        trainset,
        batch_size=hparams.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=8,                   # ↑
        pin_memory=True,                 # ↑
        persistent_workers=True,         # ↑
        prefetch_factor=2,               # ↑
        drop_last=True,
        collate_fn=collate_fn,
        timeout=0,
    )
    return train_loader, valset, collate_fn

# def prepare_dataloaders(hparams):
#     # Get data, data loaders and collate function ready
#     trainset = TextMelLoader(hparams.training_files, hparams)
#     valset = TextMelLoader(hparams.validation_files, hparams)
#     collate_fn = TextMelCollate(hparams.n_frames_per_step)

#     if hparams.distributed_run:
#         train_sampler = DistributedSampler(trainset)
#         shuffle = False
#     else:
#         train_sampler = None
#         shuffle = True

#     train_loader = DataLoader(trainset, num_workers=1, shuffle=shuffle,
#                               sampler=train_sampler,
#                               batch_size=hparams.batch_size, pin_memory=False,
#                               drop_last=True, collate_fn=collate_fn)
#     return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def load_model(hparams):
    model = Tacotron2(hparams)

    if torch.cuda.is_available():
        model = model.cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model

def warm_start_model(checkpoint_path, model, ignore_layers):
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")  # 멀티GPU 안전
    except Exception:
        ckpt = torch.load(checkpoint_path, map_location="cpu")

    sd = ckpt.get('state_dict', ckpt)

    # prefix 정리 (DataParallel)
    sd = { (k[7:] if k.startswith('module.') else k): v for k, v in sd.items() }

    # 삭제/무시 처리는 sd에서
    ignore = set(ignore_layers or [])
    for k in list(sd.keys()):
        if k in ignore:
            sd.pop(k)

    # 추가로 임베딩 강제 삭제가 필요하면 여기서
    for key in ['embedding.weight']:
        if key in sd:
            sd.pop(key)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"warm-start missing:{missing} unexpected:{unexpected}")
    model = model.to("cuda:0" if torch.cuda.is_available() else "cpu")
    return model

# def warm_start_model(checkpoint_path, model, ignore_layers):
#     # assert os.path.isfile(checkpoint_path)
#     # print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
#     # checkpoint_dict = torch.load(checkpoint_path,weights_only=False)
#     # model_dict = checkpoint_dict['state_dict']
#     # if len(ignore_layers) > 0:
#     #     model_dict = {k: v for k, v in model_dict.items()
#     #                   if k not in ignore_layers}
#     #     dummy_dict = model.state_dict()
#     #     dummy_dict.update(model_dict)
#     #     model_dict = dummy_dict
#     # model.load_state_dict(model_dict)
#     # return model
#     try:
#         ckpt = torch.load(checkpoint_path, map_location="cuda:0", weights_only=True)
#     except Exception:
#         ckpt = torch.load(checkpoint_path, map_location="cuda:0", weights_only=False)

#     sd = ckpt.get('state_dict', ckpt)
#     state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

#     #sd = { (k[7:] if k.startswith('module.') else k): v for k,v in sd.items() }

#     # 무시/형상 불일치 제거
#     # ignore = set(ignore_layers or [])
#     # msd = model.state_dict()
#     # for k in list(sd.keys()):
#     #     if k in ignore:
#     #         sd.pop(k); continue
#     #     if k in msd and msd[k].shape != sd[k].shape:
#     #         print(f'[skip] shape mismatch: {k} {tuple(sd[k].shape)} != {tuple(msd[k].shape)}')
#     #         sd.pop(k)
#     if 'embedding.weight' in state_dict:
#       del state_dict['embedding.weight']
#     elif 'module.embedding.weight' in state_dict: # DataParallel 사용 시 접두사 확인
#       del state_dict['module.embedding.weight']

#     model.to("cuda:0")
#     ins = model.load_state_dict(sd, strict=False)
#     print(ins)
#     return model

def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=2,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        MAX_VAL_BATCHES = 20
        from torch.amp import autocast
        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            if i >= MAX_VAL_BATCHES: break
            x, y = model.parse_batch(batch)
            with autocast("cuda", enabled=True):
                y_pred = model(x)
                loss = criterion(y_pred, y)
            reduced_val_loss = (reduce_tensor(loss.detach(), n_gpus).item()
                                if distributed_run else float(loss.detach().item()))
            val_loss += reduced_val_loss
        val_loss = val_loss / max(1, min(len(val_loader), MAX_VAL_BATCHES))

    model.train()
    if rank == 0:
        print(f"Validation loss {iteration}: {val_loss:9f}")
        logger.log_validation(val_loss, model, y, y_pred, iteration)

    #     val_loss = 0.0
    #     for i, batch in enumerate(val_loader):
    #         x, y = model.parse_batch(batch)
    #         y_pred = model(x)
    #         loss = criterion(y_pred, y)
    #         if distributed_run:
    #             reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
    #         else:
    #             reduced_val_loss = loss.item()
    #         val_loss += reduced_val_loss
    #     val_loss = val_loss / (i + 1)

    # model.train()
    # if rank == 0:
    #     print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
    #     logger.log_validation(val_loss, model, y, y_pred, iteration)


def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.AdamW([
    {"params": model.embedding.parameters(), "lr": 2e-4},
    {"params": (p for n,p in model.named_parameters() if not n.startswith("embedding")), "lr": learning_rate},
], betas=(0.9, 0.98), weight_decay=hparams.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
    #                              weight_decay=hparams.weight_decay)

    # if hparams.fp16_run:
    #     from apex import amp
    #     model, optimizer = amp.initialize(
    #         model, optimizer, opt_level='O2')
    from torch.amp import GradScaler, autocast
    scaler = GradScaler("cuda", enabled=True)

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    criterion = Tacotron2Loss()

    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))
    print("##############",hparams.iters_per_checkpoint, hparams.epochs)
    model.train()
    is_overflow = False
    file_name = hparams.training_files.split("/")[-1]
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
      print(f"Epoch: {epoch}")
      if hparams.distributed_run and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
        train_loader.sampler.set_epoch(epoch)

      for i, batch in enumerate(train_loader):
          start = time.perf_counter()

          # (1) LR 갱신
          for pg in optimizer.param_groups:
              pg['lr'] = learning_rate

          # (2) zero_grad
          optimizer.zero_grad(set_to_none=True)

          # (3) 배치 파싱
          x, y = model.parse_batch(batch)

          # (4) 순전파 + 손실 (AMP)
          with autocast("cuda", enabled=True):
              y_pred = model(x)
              loss = criterion(y_pred, y)

          # (5) 로깅용 loss (DDP면 allreduce)
          if hparams.distributed_run:
              reduced_loss = reduce_tensor(loss.detach(), n_gpus)
              reduced_loss = float(reduced_loss.item())
          else:
              reduced_loss = float(loss.detach().item())

          # (6) 역전파 (스케일된 그래디언트)
          scaler.scale(loss).backward()

          # (7) grad clip은 unscale 이후!
          scaler.unscale_(optimizer)
          grad_norm = torch.nn.utils.clip_grad_norm_(
              model.parameters(), hparams.grad_clip_thresh
          )

          # (8) 옵티마이저 스텝 + 스케일러 업데이트
          # scaler.step(optimizer)
          # # scaler.update()
          # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)

          if not torch.isfinite(grad_norm):
            if rank == 0:
                print(f"[warn] skip step {iteration} (grad_norm={grad_norm})")
            optimizer.zero_grad(set_to_none=True)
            scaler.update()  # 스케일만 갱신
            iteration += 1
            continue

          scaler.step(optimizer)
          scaler.update()

          # (9) 로깅
          if rank == 0:
              duration = time.perf_counter() - start
              print(f"Train loss {iteration} {reduced_loss:.6f} "
                    f"Grad Norm {grad_norm:.6f} {duration:.2f}s/it")
              logger.log_training(reduced_loss, grad_norm, learning_rate, duration, iteration)

          # (10) 검증/체크포인트
          if (iteration % hparams.iters_per_checkpoint == 0):
              validate(model, criterion, valset, iteration,
                      hparams.batch_size, n_gpus, collate_fn, logger,
                      hparams.distributed_run, rank)
              if rank == 0:
                  checkpoint_path = os.path.join(
                      output_directory, f"checkpoint_{file_name}_{iteration}"
                  )
                  save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path)

          iteration += 1
    # for epoch in range(epoch_offset, hparams.epochs):
    #     print("Epoch: {}".format(epoch))
    #     for i, batch in enumerate(train_loader):
    #         start = time.perf_counter()
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] = learning_rate

    #         model.zero_grad()
    #         x, y = model.parse_batch(batch)
    #         with autocast(enabled=amp_enabled):
    #           y_pred = model(x)
    #           loss = criterion(y_pred, y)
    #         # y_pred = model(x)

    #         # loss = criterion(y_pred, y)
    #         # if hparams.distributed_run:
    #         #     reduced_loss = reduce_tensor(loss.data, n_gpus).item()
    #         # else:
    #         #     reduced_loss = loss.item()
    #         # if hparams.fp16_run:
    #         #     with amp.scale_loss(loss, optimizer) as scaled_loss:
    #         #         scaled_loss.backward()
    #         # else:
    #         #     loss.backward()
    #         if hparams.distributed_run:
    #             reduced_loss = reduce_tensor(loss.detach(), n_gpus)
    #             reduced_loss = float(reduced_loss.item())
    #         else:
    #             reduced_loss = float(loss.detach().item())
    #         # if hparams.fp16_run:
    #         #     grad_norm = torch.nn.utils.clip_grad_norm_(
    #         #         amp.master_params(optimizer), hparams.grad_clip_thresh)
    #         #     is_overflow = math.isnan(grad_norm)
    #         # else:
    #         #     grad_norm = torch.nn.utils.clip_grad_norm_(
    #         #         model.parameters(), hparams.grad_clip_thresh)

    #         optimizer.step()

    #         if not is_overflow and rank == 0:
    #             duration = time.perf_counter() - start
    #             print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
    #                 iteration, reduced_loss, grad_norm, duration))
    #             logger.log_training(
    #                 reduced_loss, grad_norm, learning_rate, duration, iteration)

    #         if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
    #             validate(model, criterion, valset, iteration,
    #                      hparams.batch_size, n_gpus, collate_fn, logger,
    #                      hparams.distributed_run, rank)
    #             if rank == 0:
    #                 checkpoint_path = os.path.join(
    #                     output_directory, "checkpoint_{}".format(iteration))
    #                 save_checkpoint(model, optimizer, learning_rate, iteration,
    #                                 checkpoint_path)

    #         iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    train(args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)
