import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import math
import copy


class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        batch_size, num_patches, embed_dim = x.size()
        height = width = int(num_patches ** 0.5)

        # Reshape x to (batch_size, embed_dim, height, width)
        x = x.view(batch_size, embed_dim, height, width)

        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)

        # Reshape back to (batch_size, num_patches, embed_dim)
        out = out.view(batch_size, num_patches, embed_dim)
        return out


class BasicLayer_up(nn.Module):
    """ A basic ResBlock layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        upsample (nn.Module | None, optional): Upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([ResBlock(dim=dim) for _ in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x, None)
        return x


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x,y):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x= self.norm(x)

        return  x #torch.Size([24, 196, 384])

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x= self.norm(x)

        return x

class Decoder(nn.Module):
    def __init__(self, embed_dim, depths, num_heads, window_size,
                 mlp_ratio, img_size, qkv_bias, qk_scale, drop_rate, attn_drop_rate, norm_layer,
                 use_checkpoint, num_layers, patch_size=4, in_chans=3,
                 drop_path_rate=0.1, patch_embed=None, patch_norm=True, final_upsample="expand_first", num_classes=1, args=None):
        super().__init__()
        self.patch_norm = patch_norm
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.num_classes = num_classes
        print("args:", args)
        self.mode = args["mode"]
        self.skip_num = args["skip_num"]
        self.operation = args["operationaddatten"]
        self.add_attention = args["spatial_attention"]
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.embed_dim = embed_dim
        num_patches = patch_embed.num_patches

        patches_resolution = patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                                         patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                       dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2,
                                       norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                                           patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths[(self.num_layers - 1 - i_layer)],
                                         norm_layer=norm_layer,
                                         upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)
        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)
        if self.final_upsample == "expand_first":
            self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size), dim_scale=4,
                                          dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"
        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = self.output(x)
        return x

    def forward(self, x, x_downsample, x_attention_encoder):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                if self.mode == "swin":
                    if self.add_attention == "1":
                        x = layer_up(x, x_attention_encoder[3 - inx])
                    else:
                        x = layer_up(x, None)
            else:
                if self.mode == "swin":
                    if self.skip_num == '1' and inx == 1:
                        x = x + x_downsample[3 - inx]
                        x = layer_up(x)
                    elif self.skip_num == '2' and inx in [1, 2]:
                        x = x + x_downsample[3 - inx]
                        x = layer_up(x)
                    elif self.skip_num == '3':
                        x = x + x_downsample[3 - inx]
                        x = layer_up(x)
                    else:
                        x = x + x_downsample[3 - inx]
                        x = layer_up(x, None)

        x = self.norm_up(x)  # B L C
        x = self.up_x4(x)
        return x
