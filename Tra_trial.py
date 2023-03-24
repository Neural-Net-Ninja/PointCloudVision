

import json






New_dict = {"unfreeze_layers": ["head_softmax"],
            "Frameworks" : [{"PointNet": [{"Initial Layers": ["conv1", "conv2", "conv3"]},
                                        {"Final Layer": ["conv4"]}],
                            "EdgeConv": [{"Initial Layers": ["conv1", "conv2", "conv3"]},
                                        {"Final Layer": ["conv4"]}],
                            "Hybrid": [{"Initial Layers": ["conv1", "conv2", "conv3"]},
                                        {"Final Layer": ["conv4"]}],
                            "RandLANet": [{"Initial Layers": ["fc_start", "bn_start", "encoders", "mlp_bottleneck", "decoders"]},
                                            {"Final Layer": ["fc_end"]}],
                            "ShellNet": [{"Initial Layers": ["shell_conv_1", "shell_conv_2", "shell_conv_3", "shell_up_3",
                                                            "shell_up_2", "shell_up_1", "fc1", "fc2"]},
                                        {"Final Layer": ["fc3"]}],
                            "KpfcnnRigid": [{"Initial Layers": ["encoder_blocks", "decoder_blocks", "head_mlp"]},
                                            {"Final Layer": ["head_softmax"]}],
                            "KpfcnnDeform": [{"Initial Layers": ["encoder_blocks", "decoder_blocks", "head_mlp"]},
                                            {"Final Layer": ["head_softmax"]}]}]}

#print(New_dict["unfreeze_layers"])


with open('X:/Run/pcnn/data/unfreeze_layers.json', "r") as f:
    Dict_unfreeze = json.load(f)['unfreeze_layers']

print('yo::::',Dict_unfreeze)