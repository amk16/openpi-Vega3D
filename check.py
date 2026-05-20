import torch
import openpi_vega3d.towers.wan.modules.attention as a
print('FA2 available:', a.FLASH_ATTN_2_AVAILABLE)
print('FA3 available:', a.FLASH_ATTN_3_AVAILABLE)
print('CUDA device:', torch.cuda.get_device_name(0))
print('CUDA capability:', torch.cuda.get_device_capability(0))
print('torch version:', torch.__version__)
