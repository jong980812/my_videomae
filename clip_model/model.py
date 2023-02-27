from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class LocalityFeedForward(nn.Module):
    def __init__(self, in_dim=768, out_dim=768, stride=1, expand_ratio=4., act='hs+se', reduction=4,
                 wo_dp_conv=False, dp_first=False):
        """
        :param in_dim: the input dimension
        :param out_dim: the output dimension. The input and output dimension should be the same.
        :param stride: stride of the depth-wise convolution.
        :param expand_ratio: expansion ratio of the hidden dimension.
        :param act: the activation function.
                    relu: ReLU
                    hs: h_swish
                    hs+se: h_swish and SE module
                    hs+eca: h_swish and ECA module
                    hs+ecah: h_swish and ECA module. Compared with eca, h_sigmoid is used.
        :param reduction: reduction rate in SE module.
        :param wo_dp_conv: without depth-wise convolution.
        :param dp_first: place depth-wise convolution as the first layer.
        """
        super(LocalityFeedForward, self).__init__()
        hidden_dim = int(in_dim * expand_ratio)
        kernel_size = 3

        layers = []
        # the first linear layer is replaced by 1x1 convolution.
        layers.extend([
            nn.Conv2d(in_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)])

        # the depth-wise convolution between the two linear layers
        if not wo_dp_conv:
            dp = [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)
            ]
            if dp_first:
                layers = dp + layers
            else:
                layers.extend(dp)

        # if act.find('+') >= 0:
        #     attn = act.split('+')[1]
        #     if attn == 'se':
        #         layers.append(SELayer(hidden_dim, reduction=reduction))
        #     elif attn.find('eca') >= 0:
        #         layers.append(ECALayer(hidden_dim, sigmoid=attn == 'eca'))
        #     else:
        #         raise NotImplementedError('Activation type {} is not implemented'.format(act))

        # the second linear layer is replaced by 1x1 convolution.
        layers.extend([
            nn.Conv2d(hidden_dim, out_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_dim)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = x + self.conv(x)
        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        if orig_type == torch.float16:
            ret = super().forward(x)
        elif orig_type == torch.float32:
            ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, ffn:str='mlp',use_time_attn :bool = False,frames:int = 1,use_adapter=False):
        super().__init__()
        """
        self.time_attn == self.attn
        self.ln_1 == self.ln_time
        """
        ########! Utility ######! 
        self.use_time_attn=use_time_attn
        self.use_adapter = use_adapter
        self.ffn=ffn
        self.frames = frames
        ########################!
        
        assert self.ffn == 'mlp' if self.use_adapter else self.ffn != None
        ########! Adapter ######! 
        if self.use_adapter:
            self.MLP_adapter = Adapter(d_model, skip_connect=False)
            self.S_adapter = Adapter(d_model)
            self.T_adapter = Adapter(d_model, skip_connect=False)
        ########################!
        
        self.ln_1 = LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.attn_mask=attn_mask
        ########! T_MSA ######! 
        if self.use_time_attn:
            self.time_attn = nn.MultiheadAttention(d_model,n_head)  
            self.ln_time = LayerNorm(d_model)
        #######################!
        self.ln_2 = LayerNorm(d_model)
        if ffn == 'mlp':
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, d_model * 4)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(d_model * 4, d_model))
            ]))
        elif ffn == 'locality':
            self.locality_ffn = LocalityFeedForward(d_model,d_model,1,4,act='hs')
        elif ffn == 'locality_v2':
            kernel_size=5
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, d_model * 4)),
                ("gelu", QuickGELU()),
                ("locality",nn.Conv2d(d_model*4, d_model*4, kernel_size=kernel_size, stride=1, padding= kernel_size// 2, groups=d_model, bias=False)),
                ("gelu_2", QuickGELU()),
                ("BN",nn.BatchNorm2d(d_model*4)),
                ("c_proj", nn.Linear(d_model * 4, d_model))
            ]))
    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    def time_attention(self, x:torch.Tensor):
        return self.time_attn(x,x,x,need_weights=False,attn_mask=None)[0]
    
    def forward(self, x):
        l,n,d=x.shape
        batch_size=n//self.frames
        if self.use_adapter:
            x = x + self.S_adapter(self.attention(self.ln_1(x)))
        else:
            x = x + self.attention(self.ln_1(x))#!Original path
        if self.use_time_attn:
            xt = rearrange(x, 'l (b t) d -> t (b l) d',t=self.frames)#!
            if self.use_adapter:
                xt =self.T_adapter(self.time_attention(self.ln_time(xt)))
            else: 
                xt =self.time_attention(self.ln_time(xt))
            xt = rearrange(xt, 't (b l) d -> l (b t) d',l=l,d=d,b=batch_size,t=self.frames)#!L,N,D
        x = x + xt
        
        if self.ffn == 'mlp':
            if self.use_adapter:
                xn = self.ln_2(x)
                x = x + self.mlp(xn) + 0.5*self.MLP_adapter(xn)
            else:
                x = x + self.mlp(self.ln_2(x))#!Original path
        elif self.ffn == 'locality':
            x = x.transpose(0,1)                                #!n,b,d --> b,n,d
            b,n,d= x.shape
            #?b: batch size n: token len d: token dim
            
            cls_token, x = torch.split(x, [1, n-1],dim=1)       #! (b,1,d), (b,196,d)
            x = x.transpose(1,2).view(b,d,14,14)                #! (b, d, 14,14)
            
            x = self.locality_ffn(x).flatten(2).transpose(1,2)  #!(b,196,d)
            x = torch.cat([cls_token, x], dim=1)                #!(b,197,d)
            x = x.transpose(0,1)                                #!(n,b,d)
        elif self.ffn == 'locality_v2':
            origin = x
            x = self.ln_2(x)
            x = self.mlp[0](x)# Linear dimi -> 4*dim
            x = self.mlp[1](x)# GELU
            x = x.transpose(0,1)                                #!n,b,d --> b,n,d
            b,n,d= x.shape
            #?b: batch size n: token len d: token dim
            cls_token, x = torch.split(x, [1, n-1],dim=1)       #! (b,1,d), (b,196,d)
            x = x.transpose(1,2).view(b,d,14,14)
            x = (x + self.mlp[4](self.mlp[3](self.mlp[2](x)))).flatten(2).transpose(1,2)
            x = torch.cat([cls_token, x], dim=1) 
            x = x.transpose(0,1)
            x = origin + self.mlp[5](x)
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, ffn: str = 'mlp', use_time_attn = False,frames = 1,use_adapter=False,module_layers:list = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.module_layers=module_layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask,ffn=ffn, use_time_attn=use_time_attn,frames=frames,use_adapter=use_adapter) if num in self.module_layers #!Additional Module Layers.
                                        else ResidualAttentionBlock(width,heads,attn_mask,ffn='mlp',use_time_attn=False,frames=frames,use_adapter=False)#! Vanila Block
                                        for num in range(layers)])

    def forward(self, x: torch.Tensor):
        for blk in self.resblocks:
            x = blk(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int,ffn: str = 'mlp',use_time_attn: bool = False,frames:int = 1, use_adapter=False,module_layers:list = None):
        super().__init__()
        self.frames=frames#!
        self.use_time_attn=use_time_attn#!
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))#197,768
        if frames != 1:#! not 1 frame.
            self.time_positional_embedding = nn.Parameter(scale * torch.randn(1,frames,1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads,ffn=ffn, use_time_attn=use_time_attn,frames=frames,use_adapter = use_adapter,module_layers=module_layers)

        self.ln_post = LayerNorm(width)
        
        self.layers = layers
        

    def forward(self, x: torch.Tensor):
        fusion = None
        if len(x.shape) == 5:
            fusion = True
        
        if fusion:
            b, t = x.shape[0], x.shape[2]
            x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        if self.use_time_attn:
            _,l,d= x.shape
            x = rearrange(x,'(b t) l d -> b t l d',b=b,t=t)#! 2, 16, 197, 768
            x = x + self.time_positional_embedding.to(x.dtype)
            x = rearrange(x,'b t l d -> (b t) l d',b=b,t=t)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        if fusion:
            x = rearrange(x, '(b t) n d -> b t n d', t=t)
            x = self.ln_post(x[:, :, 0, :])
            x = x.mean(dim=1)
        else:
            x = self.ln_post(x[:, 0, :])
        

        return x


class CLIP(nn.Module):
    def __init__(self,
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 ffn: str = 'mlp',
                 nb_classes:int = 300,
                 time_attn = False,
                 frames:int = 1,
                 adapter = False,
                 module_layers:list = None
                 ):
        super().__init__()
        self.patch_size=vision_patch_size
        vision_heads = vision_width // 64
        self.layers = vision_layers
        self.use_adapter = adapter#!!!!!!!!!!!using AIM adapter
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            ffn=ffn,
            use_time_attn=time_attn,
            frames=frames,
            use_adapter = adapter,
            module_layers=module_layers)
        
        self.head = nn.Linear(vision_width, nb_classes) # 수동으로 class 수 맞춰줘야함 load엑서 변수가 통제되어있음

        self.initialize_parameters()
        
    def get_num_layers(self):
        return self.layers
    
    def no_weight_decay(self):
        return {}

    def initialize_parameters(self):

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)
        if self.use_adapter:
        ##!!!!!!!!!!!!initialize S_Adapter
            for n, m in self.visual.transformer.named_modules():
                if 'S_adapter' in n:
                    for n2, m2 in m.named_modules():
                        if 'D_fc2' in n2:
                            if isinstance(m2, nn.Linear):
                                nn.init.constant_(m2.weight, 0)
                                nn.init.constant_(m2.bias, 0)

            ## initialize T_Adapter
            for n, m in self.visual.transformer.named_modules():
                if 'T_adapter' in n:
                    for n2, m2 in m.named_modules():
                        if 'D_fc2' in n2:
                            if isinstance(m2, nn.Linear):
                                nn.init.constant_(m2.weight, 0)
                                nn.init.constant_(m2.bias, 0)

            ## initialize MLP_Adapter
            for n, m in self.visual.transformer.named_modules():
                if 'MLP_adapter' in n:
                    for n2, m2 in m.named_modules():
                        if 'D_fc2' in n2:
                            if isinstance(m2, nn.Linear):
                                nn.init.constant_(m2.weight, 0)
                                nn.init.constant_(m2.bias, 0)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, x):
        return self.visual(x)


    def forward(self, x):
        x = self.encode_image(x)
        x = self.head(x)
        
        return x


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn. Linear,nn.BatchNorm2d)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict,args=None):
    
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
    #!!!!!!!!!!!!!!!!!!!!!!!!
    if args is not None:
        ffn = args.ffn
        nb_classes=args.nb_classes
        use_clip_time_attn = args.use_clip_time_attn
        num_frames=args.num_frames
        use_adapter = args.use_adapter
        module_layers=args.module_layers
        print(module_layers)
    else:#!My hard coding
        ffn = 'locality'
        nb_classes=300
    #!!!!!!!!!!!!!!!!!!!!!!!!!!

    model = CLIP(
        image_resolution, vision_layers, vision_width, vision_patch_size,
        ffn=ffn,
        nb_classes=nb_classes, 
        time_attn=use_clip_time_attn, 
        frames=num_frames,
        adapter = use_adapter, 
        module_layers=module_layers
        )


    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict, strict=False)
    if args.use_clip_time_attn:
        with torch.no_grad():#! module_layers에 들어가는 것만 한다. 
            for i in module_layers:
                model.visual.transformer.resblocks[i].time_attn.out_proj.weight.copy_(model.visual.transformer.resblocks[i].attn.out_proj.weight)
                model.visual.transformer.resblocks[i].time_attn.out_proj.bias.copy_(model.visual.transformer.resblocks[i].attn.out_proj.bias)
                model.visual.transformer.resblocks[i].ln_time.weight.copy_(model.visual.transformer.resblocks[i].ln_1.weight)
                model.visual.transformer.resblocks[i].ln_time.bias.copy_(model.visual.transformer.resblocks[i].ln_1.bias)
        print("Time Attention module initialize same with self.attn")
    return model
