import torch
from train import MusicTransformer, device
from tokenizer import sos_id, decode

# After using torch.compile, the model is saved with a prefix '_orig_mod.' in the state_dict keys.
# This function removes the prefix from the keys.
def remove_prefix_from_state_dict(state_dict, prefix='_orig_mod.'):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

model = MusicTransformer()
checkpoint_path = 'checkpoint.pt'
checkpoint = torch.load(checkpoint_path)
model = model.to(device)
model.load_state_dict(remove_prefix_from_state_dict(checkpoint['model_state_dict']))
model.eval()

tokens = model.sample(sos_id, 2000, temp=1.0, topk=30)[0, 1:].tolist()
decode(tokens, "checkpoint.mid")