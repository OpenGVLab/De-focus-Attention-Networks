import torch

ckpt_path = 'path/to/cls/ckpt'
# Load the original checkpoint
ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))

# Create a new dictionary to store modified parameters
new_ckpt = {}

def build_mapping():
    mapping = {
        'patch_embed.proj.weight': 'patch_embed.projection.weight',
        'patch_embed.proj.bias': 'patch_embed.projection.bias',
    }

    return mapping

mapping = build_mapping()

# Add prefix to each key in the state_dict
for key, value in ckpt['model'].items():
    new_name = mapping.get(key)
    if new_name is None:
        new_name = key

    new_ckpt[new_name] = value

# Save the modified checkpoint
new_ckpt_path = 'path/to/converted/ckpt'
torch.save(new_ckpt, new_ckpt_path)