import torch

ckpt_path = 'path/to/22k/ckpt'
# Load the original checkpoint
ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))

# Create a new dictionary to store modified parameters
new_ckpt = {}

# extract 1k
convert_file = './classification/data/map22kto1k.txt'
with open(convert_file, 'r') as f:
    convert_list = [int(line) for line in f.readlines()]

# Add prefix to each key in the state_dict
for key, value in ckpt['model_ema'].items():
    if key == 'head.2.bias':
        new_ckpt[key] = value[convert_list]
    elif key == 'head.2.weight':
        new_ckpt[key] = value[convert_list]
    elif key == 'head2.2.bias':
        new_ckpt[key] = value[convert_list]
    elif key == 'head2.2.weight':
        new_ckpt[key] = value[convert_list]
    else:
        new_ckpt[key] = value

# Save the modified checkpoint
new_ckpt_path = 'path/to/converted/ckpt'
torch.save({'model': new_ckpt, 'model_ema': new_ckpt}, new_ckpt_path)
