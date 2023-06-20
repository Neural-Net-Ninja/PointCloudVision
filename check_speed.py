import torch
import time

# start timer
t0 = time.time()

a = torch.randn(1000000000)

mul = torch.mul(a, a)
# end timer
t1 = time.time()

elapsed_time = (t1 - t0) / 60
# round to 2 decimal places
elapsed_time = round(elapsed_time, 2)
print('Elapsed time:', elapsed_time, 'minutes')
