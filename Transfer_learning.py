import torch

checkpoint_path = r'X:/architectures/randla-net-baseline-0/randla-net-baseline-0_best.pth'
model =RandLANet(3, 3)

# original saved file with DataParallel
state_dict = torch.load(checkpoint_path)
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k # remove `module.`
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)



print(model['fc_end'])