from collections import OrderedDict
import torch


def convert_videomae_pretrain(path):
    old_ckpts = torch.load(path, map_location='cpu')
    new_ckpts = OrderedDict()

    for k, v in old_ckpts['model'].items():
        if not k.startswith('encoder.'):
            continue
        if k.startswith('encoder.blocks.'):
            spk = k.split('.')
            if '.'.join(spk[3:]) == 'attn.qkv.weight':
                assert v.shape[0] % 3 == 0, v.shape
                qi, ki, vi = torch.split(v, v.shape[0] // 3, dim=0)
                new_ckpts['.'.join(spk[:3] + ['attn', 'q_proj', 'weight'])] = qi
                new_ckpts['.'.join(spk[:3] + ['attn', 'k_proj', 'weight'])] = ki
                new_ckpts['.'.join(spk[:3] + ['attn', 'v_proj', 'weight'])] = vi
            elif '.'.join(spk[3:]) == 'mlp.fc1.bias':  # 'blocks.1.norm1.weight' --> 'norm1.weight'
                new_ckpts['.'.join(spk[:3] + ['fc1', 'bias'])] = v
            elif '.'.join(spk[3:]) == 'mlp.fc1.weight':
                new_ckpts['.'.join(spk[:3] + ['fc1', 'weight'])] = v
            elif '.'.join(spk[3:]) == 'mlp.fc2.bias':
                new_ckpts['.'.join(spk[:3] + ['fc2', 'bias'])] = v
            elif '.'.join(spk[3:]) == 'mlp.fc2.weight':
                new_ckpts['.'.join(spk[:3] + ['fc2', 'weight'])] = v
            else:
                new_ckpts[k] = v
        else:
            new_ckpts[k] = v

    assert path.endswith('.pth'), path
    new_path = path[:-4] + '_new.pth'
    torch.save(OrderedDict(model=new_ckpts), new_path)
    print('Finished :', path)

if __name__ == '__main__':
    path = '/path/to/videomae/pretrained/checkpoint.pth'
    convert_videomae_pretrain(path)
