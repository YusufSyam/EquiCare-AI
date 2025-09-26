import torch
print(torch.cuda.is_available())   # Harus True
print(torch.cuda.device_count())   # Harus > 0
print(torch.cuda.get_device_name(0))  # Nama GPU
