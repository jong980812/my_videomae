from collections import OrderedDict
from typing import Tuple, Union
from timm.models.layers import DropPath,to_2tuple, trunc_normal_
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
    def __init__(self, d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None,
                 ffn:str='mlp',
                 use_time_attn :bool = False,
                 frames:int = 1,
                 use_adapter=False,
                 adapter_scale :float = 0.5,
                 num_t_adapter:int = 1,
                 drop_path = 0.):
        super().__init__()
        """
        self.time_attn == self.attn
        self.ln_1 == self.ln_time
        """
        ########! Utility ######! 
        self.use_time_attn = use_time_attn
        self.use_adapter = use_adapter
        self.ffn = ffn
        self.frames = frames
        self.adapter_scale = adapter_scale
        self.num_t_adapter = num_t_adapter
        ########################!
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        assert self.ffn == 'mlp' if self.use_adapter else self.ffn != None
        ########! Adapter ######! 
        if self.use_adapter:
            self.MLP_adapter = Adapter(d_model, skip_connect=False)
            self.S_adapter = Adapter(d_model)
            self.T_adapter = Adapter(d_model, skip_connect=False)
        if self.num_t_adapter == 2:
            self.T_adapter_in = Adapter(d_model)
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
                if self.num_t_adapter == 2:
                    xt =self.T_adapter(self.time_attention(self.T_adapter_in(self.ln_time(xt))))
                else:
                    xt =self.T_adapter(self.time_attention(self.ln_time(xt)))
            else: 
                xt =self.time_attention(self.ln_time(xt))
            xt = rearrange(xt, 't (b l) d -> l (b t) d',l=l,d=d,b=batch_size,t=self.frames)#!L,N,D
        x = x + xt
        
        if self.ffn == 'mlp':
            if self.use_adapter:
                xn = self.ln_2(x)
                x = x + self.mlp(xn) + self.adapter_scale*self.MLP_adapter(xn)
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
    def __init__(self,  width: int,
                        layers: int, 
                        heads: int, 
                        attn_mask: torch.Tensor = None, 
                        ffn: str = 'mlp', 
                        use_time_attn = False,
                        frames = 1,
                        use_adapter=False,
                        module_layers:list = None,
                        adapter_scale :float= 0.5,
                        num_t_adapter:int=1,
                        drop_path = 0.):
        super().__init__()
        self.width = width
        self.layers = layers
        self.module_layers=module_layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask,ffn=ffn, use_time_attn=use_time_attn,frames=frames,use_adapter=use_adapter,adapter_scale=adapter_scale,num_t_adapter=num_t_adapter,drop_path = 0.) if num in self.module_layers #!Additional Module Layers.
                                        else ResidualAttentionBlock(width,heads,attn_mask,ffn='mlp',use_time_attn=False,frames=frames,use_adapter=False)#! Vanila Block
                                        for num in range(layers)])

    def forward(self, x: torch.Tensor):
        for blk in self.resblocks:
            x = blk(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 input_resolution: int, 
                 patch_size: int, 
                 width: int, 
                 layers: int, 
                 heads: int,
                 ffn: str = 'mlp',
                 use_time_attn: bool = False,
                 frames:int = 1, 
                 use_adapter=False,
                 module_layers:list = None,
                 adapter_scale :float= 0.5,
                 num_t_adapter:int=1, 
                 drop_path = 0.):
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

        self.transformer = Transformer(width, 
                                       layers,
                                       heads,
                                       ffn=ffn,
                                       use_time_attn=self.use_time_attn,
                                       frames=self.frames,
                                       use_adapter = use_adapter,
                                       module_layers=module_layers,
                                       adapter_scale=adapter_scale,
                                       num_t_adapter=num_t_adapter,
                                       drop_path=drop_path)

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
                 module_layers:list = None,
                 adapter_scale :float= 0.5,
                 num_t_adapter:int=1,
                 time_attn_random:bool = False,
                 ):
        super().__init__()
        self.patch_size=vision_patch_size
        vision_heads = vision_width // 64
        self.layers = vision_layers
        self.use_adapter = adapter#!!!!!!!!!!!using AIM adapter
        self.time_attn_random = time_attn_random
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
            module_layers=module_layers,
            adapter_scale =adapter_scale,
            num_t_adapter=num_t_adapter)
        
        self.head = nn.Linear(vision_width, nb_classes) # 수동으로 class 수 맞춰줘야함 load엑서 변수가 통제되어있음

        self.initialize_parameters()
        
    def get_num_layers(self):
        return self.layers
    
    def no_weight_decay(self):
        return {}

    def initialize_parameters(self):

        # if isinstance(self.visual, ModifiedResNet):
        #     if self.visual.attnpool is not None:
        #         std = self.visual.attnpool.c_proj.in_features ** -0.5
        #         nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
        #         nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
        #         nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
        #         nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)
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
        proj_std = (self.visual.transformer.width ** -0.5) * ((2 * self.visual.transformer.layers) ** -0.5)
        attn_std = self.visual.transformer.width ** -0.5
        fc_std = (2 * self.visual.transformer.width) ** -0.5
        if self.time_attn_random:
           for block in self.visual.transformer.resblocks:
                nn.init.normal_(block.time_attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.time_attn.out_proj.weight, std=proj_std)
                # nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                # nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        print("Weight Initialize")
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
        adapter_scale = args.adapter_scale
        num_t_adapter = args.num_t_adapter
        time_attn_random = args.time_attn_random
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
        module_layers=module_layers,
        adapter_scale=adapter_scale,
        num_t_adapter=num_t_adapter,
        time_attn_random = time_attn_random,
        )


    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict, strict=False)
    if args.use_clip_time_attn and not args.time_attn_random:
        with torch.no_grad():#! module_layers에 들어가는 것만 한다. 
            for i in module_layers:
                model.visual.transformer.resblocks[i].time_attn.out_proj.weight.copy_(model.visual.transformer.resblocks[i].attn.out_proj.weight)
                model.visual.transformer.resblocks[i].time_attn.out_proj.bias.copy_(model.visual.transformer.resblocks[i].attn.out_proj.bias)
                model.visual.transformer.resblocks[i].time_attn.in_proj_weight.copy_(model.visual.transformer.resblocks[i].attn.in_proj_weight)
                model.visual.transformer.resblocks[i].time_attn.in_proj_bias.copy_(model.visual.transformer.resblocks[i].attn.in_proj_bias)
                model.visual.transformer.resblocks[i].ln_time.weight.copy_(model.visual.transformer.resblocks[i].ln_1.weight)
                model.visual.transformer.resblocks[i].ln_time.bias.copy_(model.visual.transformer.resblocks[i].ln_1.bias)
        print("Time Attention module copy with self.attn")
    return model
