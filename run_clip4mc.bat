@echo off
set NCCL_DEBUG=info
set OMP_NUM_THREADS=1

torchrun --nproc_per_node=1 train_ddp_clip4mc.py --use_mask --batch_size 1 --batch_size_test 1 