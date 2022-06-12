## Prepare Pre-trained Checkpoints


### Video
We directly use existing pre-trained model. For video,
we use Kinetics-400 pre-trained model by [VideoMAE](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md#kinetics-400) for 1600 epochs.

We made two minor modifications to the implementation of the ViT for flexible experiments:
* We unbind the queue, key, value linear projection layer to three linear layers:
```python
# From:
self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)

# To:
self.q_proj = nn.Linear(dim, all_head_dim, bias=False)
self.v_proj = nn.Linear(dim, all_head_dim, bias=False)
self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
```
* We decompose the encapsulated `Mlp` block:
```python
# From:
self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

# To:
self.fc1 = nn.Linear(dim, mlp_hidden_dim)
self.fc2 = nn.Linear(mlp_hidden_dim, dim)
self.act = act_layer()
self.mlp_drop = nn.Dropout(drop)
```


Therefore, we need to convert the checkpoint provided by [VideoMAE](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md#kinetics-400) using [convert.py](convert.py).

Or you can directly use our preprocessed [videomae_pretrain_vit_b_1600.pth](https://github.com/ShoufaChen/AdaptFormer/releases/download/v0.1/videomae_pretrain_vit_b_1600.pth).



### Image
[mae_pretrain_vit_b.pth](https://github.com/ShoufaChen/AdaptFormer/releases/download/v0.1/mae_pretrain_vit_b.pth)