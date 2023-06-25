import torch
from models import *
from models.KPFCNN import Kpfcnn, KpfcnnRigid, KpfcnnDeform


# checkpoint_path = r'X:/architectures/EdgeConv_best_optim/EdgeConv_best.pth'
# model = EdgeConv(15, 3)

# checkpoint_path = r'K:/Autobahn1/000_Vorlage/000000______Model/Hybrid_best.pth'
# model = Hybrid(10, 4)

# checkpoint_path = r'X:/architectures/pointnet-baseline-0/pointnet-baseline-0_best.pth'
# model = PointNet(3, 4)

# checkpoint_path = r'X:/architectures/shellnet-baseline-0/shellnet-baseline-0_best.pth'
# model = ShellNet(5, 4)

# checkpoint_path = r'X:/architectures/kpfcnn-rigid-baseline-0/kpfcnn-rigid-baseline-0_best.pth'
# model = KpfcnnRigid(5,3)

checkpoint_path = r'X:/architectures/kpfcnn-deform-baseline-0/kpfcnn-deform-baseline-0_best.pth'
model = KpfcnnDeform(5,3)

# original saved file with DataParallel
state_dict = torch.load(checkpoint_path)
#print('state_dict::::', model.state_dict(state_dict))

# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    #print('key:', k)
    layer_name = k.split(".")[0]
    name = k[7:] if k.startswith('module.') else k # remove `module.`
    #print('layer_name', name)
    new_state_dict[name] = v
#print('new_state_dict.keys()', new_state_dict.keys())
# load params
model.load_state_dict(new_state_dict)
# Am Freitag habe ich eine Funktion entwickelt, um Kandidaten aus einem aktiven Punkt selektieren (developed).
# Poisson-disk sampling .
for k, v in state_dict.items():
    #print('key:', k)
    layer_name = k.split(".")[0]
    name = k[7:] if k.startswith('module.') else k # remove `module.`
    #print('layer_name', name)
    new_state_dict[name] = v
# Diese Funktion berechnet (calculated) die Distanz jedes Punktes zum aktive point und eliminiert Points, 
# die zu weit oder zu nah sind. 

# Diese selected candidate werden in die Liste der selected sample aufgenommen (recorded),
# und unter den selected candidate,,,, wird eine ramdom sample als nächster active point ausgewählt. 
print([name for name, child in model.named_children()])

print('model parameter :', model.parameters())

# Diese loop wird so lange fortgesetzt (continued)........, bis der sample size erreicht ist (reached).

# i will check its corrcetness and request a review.

# Check if the last layer is empty
last_layer = [child for name, child in self.named_children() if name==layers[-1]]
self.assertFalse(last_layer)

# Außerdem werde ich die prediction für die Essen Daten ausführen (carry out)

# und es wird eine meeting mit Justus über die Trainings praktiken geben, um das Training uniform zu halten.


specific_naemd = model.named_children(['fc_end'])
print('specific_child::::::', specific_naemd)

print([name for name, child in model.named_children()])

print('model parameter :', model.parameters())
#print('model eval :', model.eval())
#print('model named_children() :', model.named_children())#

#print('model Name******** :', [model.__class__.__name__] )

for name, child in model.named_parameters():
    print('name::::', name, '\n'
          'child::::', child)

#     # unfreeze the selected layers for fine-tuning
#     for param in child.parameters():
#         print('param.requires_grad :', param.requires_grad)
# unfreeze_layers = ['fc_end']
# line= [param.requires_grad for name, child in model.named_children() if name in unfreeze_layers
#                         for param in child.parameters()]
# print(line)


# framework_dict = {'PointNet': {{'Initial Layers': ['conv1', 'conv2', 'conv3']},
#                                {'Final Layer': ['conv4']}},
#                   'EdgeConv': {{'Initial Layers': ['conv1', 'conv2', 'conv3']},
#                                {'Final Layer': ['conv4']}},
#                   'Hybrid': {{'Initial Layers': ['conv1', 'conv2', 'conv3']},
#                              {'Final Layer': ['conv4']}},
#                   'RandLANet': {{'Initial Layers': ['fc_start', 'bn_start', 'encoders', 'mlp_bottleneck', 'decoders']},
#                                {'Final Layer': ['fc_end']}},
#                   'ShellNet': {{'Initial Layers': ['shell_conv_1', 'shell_conv_2', 'shell_conv_3', 'shell_up_3',
#                                                   'shell_up_2', 'shell_up_1', 'fc1', 'fc2']},
#                                {'Final Layer': ['fc3']}},
#                   'KpfcnnRigid': {{'Initial Layers': ['encoder_blocks', 'decoder_blocks', 'head_mlp']},
#                                   {'Final Layer': ['head_softmax']}},
#                   'KpfcnnDeform': {{'Initial Layers': ['encoder_blocks', 'decoder_blocks', 'head_mlp']},
#                                   {'Final Layer': ['head_softmax']}}}


# # Check if the last layer is empty
# last_layer = [child for name, child in self.named_children() if name==layers[-1]]
# self.assertFalse(last_layer)

# # check if only the specific layers are frozen
# frozen_layer = all([param.requires_grad for name, child in self.named_children() if name not in unfreeze_layers
#                     for param in child.parameters()])
# self.assertFalse(frozen_layer)