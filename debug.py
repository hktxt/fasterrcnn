import torch
import gc


print('current memory allocated: {}'.format(torch.cuda.memory_allocated() / 1024 ** 2))
print('max memory allocated: {}'.format(torch.cuda.max_memory_allocated() / 1024 ** 2))
print('cached memory: {}'.format(torch.cuda.memory_cached() / 1024 ** 2))
for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(type(obj), obj.size())
    except: pass