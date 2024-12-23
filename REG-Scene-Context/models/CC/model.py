import torch
import torch.nn as nn
import clip
from torch.nn import functional as nnf
from enum import Enum
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import argparse
import json
from typing import Tuple, Optional, Union

device = "cuda" if torch.cuda.is_available() else "cpu"


class MappingType(Enum):
    MLP = "mlp"
    Transformer = "transformer"


class CLIP_Backbone(nn.Module):
    def __init__(
        self, clip_model_type="ViT-B/32", normalize_prefix=False, device=device
    ):
        super(CLIP_Backbone, self).__init__()
        self.device = device
        self.normalize_prefix = normalize_prefix
        self.clip_model_name = clip_model_type.replace("/", "_")
        self.clip_model, self.preprocess = clip.load(
            clip_model_type, device=device, jit=False
        )

    def forward(self, image, from_raw: bool = False):
        if from_raw:
            image = self.preprocess(image).unsqueeze(0).to(self.device)
        prefix = self.clip_model.encode_image(
            image
        ).float()  # convert prefix to float32
        if self.normalize_prefix:
            prefix = prefix / prefix.norm(2, -1)
        return prefix


class MLP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class MlpTransformer(nn.Module):
    def __init__(
        self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.0
    ):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim**-0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(
            b, m, 2, self.num_heads, c // self.num_heads
        )
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum("bnhd,bmhd->bnmh", queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum("bnmh,bmhd->bnhd", attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):
    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(
        self,
        dim_self,
        dim_ref,
        num_heads,
        mlp_ratio=4.0,
        bias=False,
        dropout=0.0,
        act=nnf.relu,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(
            dim_self, dim_ref, num_heads, bias=bias, dropout=dropout
        )
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(
            dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout
        )


class Transformer(nn.Module):
    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec:  # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(
        self,
        dim_self: int,
        num_heads: int,
        num_layers: int,
        dim_ref: Optional[int] = None,
        mlp_ratio: float = 2.0,
        act=nnf.relu,
        norm_layer: nn.Module = nn.LayerNorm,
        enc_dec: bool = False,
    ):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(
                    TransformerLayer(
                        dim_self,
                        dim_ref,
                        num_heads,
                        mlp_ratio,
                        act=act,
                        norm_layer=norm_layer,
                    )
                )
            elif enc_dec:  # self
                layers.append(
                    TransformerLayer(
                        dim_self,
                        dim_self,
                        num_heads,
                        mlp_ratio,
                        act=act,
                        norm_layer=norm_layer,
                    )
                )
            else:  # self or cross
                layers.append(
                    TransformerLayer(
                        dim_self,
                        dim_ref,
                        num_heads,
                        mlp_ratio,
                        act=act,
                        norm_layer=norm_layer,
                    )
                )
        self.layers = nn.ModuleList(layers)


class TransformerMapper(nn.Module):
    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(
            x.shape[0], *self.prefix_const.shape
        )
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(
        self,
        dim_clip: int,
        dim_embedding: int,
        prefix_length: int,
        clip_length: int,
        num_layers: int = 8,
    ):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(
            torch.randn(prefix_length, dim_embedding), requires_grad=True
        )


class ClipREGModel(nn.Module):
    def __init__(
        self,
        prefix_length: int,
        clip_length: Optional[int] = None,
        prefix_size: int = 512,
        num_layers: int = 8,
        mapping_type: MappingType = MappingType.MLP,
        clip_model_type="ViT-B/32",
        device=device,
    ):
        super(ClipREGModel, self).__init__()
        self.device = device
        prefix_length = prefix_length
        self.prefix_length = prefix_length
        assert (
            prefix_length - 1
        ) % 2 == 0, "invalid prefix length: prefix_length-1 must be divisible by 2"
        mapping_prefix_length = (prefix_length - 1) // 2  # 1 for location features

        print(
            f"total prefix length: {prefix_length}, target/context prefix: {mapping_prefix_length}"
        )

        self.mapping_prefix_length = mapping_prefix_length
        self.backbone = CLIP_Backbone(clip_model_type=clip_model_type, device=device)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if mapping_type == MappingType.MLP:
            self.target_clip_project = MLP(
                (
                    prefix_size,
                    (self.gpt_embedding_size * mapping_prefix_length) // 2,
                    self.gpt_embedding_size * mapping_prefix_length,
                )
            )
            self.context_clip_project = MLP(
                (
                    prefix_size,
                    (self.gpt_embedding_size * mapping_prefix_length) // 2,
                    self.gpt_embedding_size * mapping_prefix_length,
                )
            )
        else:
            self.target_clip_project = TransformerMapper(
                prefix_size,
                self.gpt_embedding_size,
                mapping_prefix_length,
                clip_length,
                num_layers,
            )
            self.context_clip_project = TransformerMapper(
                prefix_size,
                self.gpt_embedding_size,
                mapping_prefix_length,
                clip_length,
                num_layers,
            )
        self.loc_project = nn.Linear(5, self.gpt_embedding_size)

    def get_dummy_token(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(
            batch_size, self.prefix_length, dtype=torch.int64, device=self.device
        )

    def make_visual_prefix(self, target, context, loc, from_raw=False):
        # target
        target_prefix = self.backbone(target, from_raw)
        target_prefix_projections = self.target_clip_project(target_prefix).view(
            -1, self.mapping_prefix_length, self.gpt_embedding_size
        )
        # context
        context_prefix = self.backbone(context, from_raw)
        context_prefix_projections = self.context_clip_project(context_prefix).view(
            -1, self.mapping_prefix_length, self.gpt_embedding_size
        )
        # loc
        loc_projections = self.loc_project(loc).unsqueeze(1)
        # Concat
        vis_prefix = torch.cat(
            (target_prefix_projections, context_prefix_projections, loc_projections),
            dim=1,
        )

        return vis_prefix

    def forward(
        self,
        tokens: torch.Tensor,
        target,
        context,
        loc,
        mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        from_raw: bool = False,
    ):
        embedding_text = self.gpt.transformer.wte(tokens)
        # target / context / loc
        vis_prefix = self.make_visual_prefix(target=target, context=context, loc=loc, from_raw=from_raw)
        # concat
        embedding_cat = torch.cat((vis_prefix, embedding_text), dim=1)

        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)

        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out


class ClipNoContextREGModel(nn.Module):
    def __init__(
        self,
        prefix_length: int,
        clip_length: Optional[int] = None,
        prefix_size: int = 512,
        num_layers: int = 8,
        mapping_type: MappingType = MappingType.MLP,
        clip_model_type="ViT-B/32",
        device=device,
    ):
        super(ClipNoContextREGModel, self).__init__()
        self.device = device
        # prefix length is reduced (to keep target prefix size equal if no context is used)
        prefix_length = (
            prefix_length // 2
        ) + 1  # half of the size since no context is used; 1 for location features
        self.prefix_length = prefix_length
        assert (
            prefix_length - 1
        ) % 2 == 0, "invalid prefix length: prefix_length-1 must be divisible by 2"
        mapping_prefix_length = prefix_length - 1

        print(
            f"total prefix length: {prefix_length}, target prefix: {mapping_prefix_length}"
        )

        self.mapping_prefix_length = mapping_prefix_length
        self.backbone = CLIP_Backbone(clip_model_type=clip_model_type, device=device)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if mapping_type == MappingType.MLP:
            self.target_clip_project = MLP(
                (
                    prefix_size,
                    (self.gpt_embedding_size * mapping_prefix_length) // 2,
                    self.gpt_embedding_size * mapping_prefix_length,
                )
            )
        else:
            self.target_clip_project = TransformerMapper(
                prefix_size,
                self.gpt_embedding_size,
                mapping_prefix_length,
                clip_length,
                num_layers,
            )
        self.loc_project = nn.Linear(5, self.gpt_embedding_size)

    def get_dummy_token(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(
            batch_size, self.prefix_length, dtype=torch.int64, device=self.device
        )

    def make_visual_prefix(self, target, loc, from_raw=False):
        # target
        target_prefix = self.backbone(target, from_raw)
        target_prefix_projections = self.target_clip_project(target_prefix).view(
            -1, self.mapping_prefix_length, self.gpt_embedding_size
        )
        # loc
        loc_projections = self.loc_project(loc).unsqueeze(1)
        # Concat
        vis_prefix = torch.cat((target_prefix_projections, loc_projections), dim=1)

        return vis_prefix

    def forward(
        self,
        tokens: torch.Tensor,
        target,
        loc,
        mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        from_raw: bool = False,
    ):
        embedding_text = self.gpt.transformer.wte(tokens)
        # target / context / loc
        vis_prefix = self.make_visual_prefix(target=target, loc=loc, from_raw=from_raw)
        # concat
        embedding_cat = torch.cat((vis_prefix, embedding_text), dim=1)

        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)

        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    
class ClipSceneREGModel(nn.Module):
    def __init__(
        self,
        prefix_length: int,
        clip_length: Optional[int] = None,
        prefix_size: int = 512,
        num_layers: int = 8,
        scene_dim: int = 134,
        mapping_type: MappingType = MappingType.MLP,
        clip_model_type="ViT-B/32",
        device=device,
    ):
        super().__init__()
        self.device = device
        prefix_length = prefix_length
        self.prefix_length = prefix_length
        assert (
            prefix_length - 1
        ) % 2 == 0, "invalid prefix length: prefix_length-1 must be divisible by 2"
        mapping_prefix_length = (prefix_length - 1) // 2  # 1 for location features

        print(
            f"total prefix length: {prefix_length}, target/context prefix: {mapping_prefix_length}"
        )

        self.scene_dim = scene_dim
        self.mapping_prefix_length = mapping_prefix_length
        self.backbone = CLIP_Backbone(clip_model_type=clip_model_type, device=device)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
                
        if mapping_type == MappingType.MLP:
            self.target_clip_project = MLP(
                (
                    prefix_size,
                    (self.gpt_embedding_size * mapping_prefix_length) // 2,
                    self.gpt_embedding_size * mapping_prefix_length,
                )
            )
            self.scene_project = MLP(
                (
                    scene_dim,
                    (self.gpt_embedding_size * mapping_prefix_length) // 2,
                    self.gpt_embedding_size * mapping_prefix_length,
                )
            )
        else:
            self.target_clip_project = TransformerMapper(
                prefix_size,
                self.gpt_embedding_size,
                mapping_prefix_length,
                clip_length,
                num_layers,
            )
            self.scene_clip_project = TransformerMapper(
                scene_dim,
                self.gpt_embedding_size,
                mapping_prefix_length,
                clip_length,
                num_layers,
            )
        self.loc_project = nn.Linear(5, self.gpt_embedding_size)

    def get_dummy_token(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(
            batch_size, self.prefix_length, dtype=torch.int64, device=self.device
        )

    def make_visual_prefix(self, target, scenesum, loc, from_raw=False):
        # target
        target_prefix = self.backbone(target, from_raw)
        target_prefix_projections = self.target_clip_project(target_prefix).view(
            -1, self.mapping_prefix_length, self.gpt_embedding_size
        )
        # scene
        scene_prefix_projections = self.scene_project(scenesum).view(
            -1, self.mapping_prefix_length, self.gpt_embedding_size
        )
        # loc
        loc_projections = self.loc_project(loc).unsqueeze(1)
        # Concat
        vis_prefix = torch.cat(
            (target_prefix_projections, scene_prefix_projections, loc_projections),
            dim=1,
        )

        return vis_prefix

    def forward(
        self,
        tokens: torch.Tensor,
        target,
        scenesum,
        loc,
        mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        from_raw: bool = False,
    ):
        embedding_text = self.gpt.transformer.wte(tokens)
        # target / context / loc
        vis_prefix = self.make_visual_prefix(target=target, scenesum=scenesum, loc=loc, from_raw=from_raw)
        # concat
        embedding_cat = torch.cat((vis_prefix, embedding_text), dim=1)

        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)

        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out


class ClipREGPrefix(ClipREGModel):
    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipREGPrefix, self).train(mode)
        self.gpt.eval()
        return self


class ClipNoContextREGPrefix(ClipREGModel):
    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipNoContextREGPrefix, self).train(mode)
        self.gpt.eval()
        return self
    
    
class ClipSceneREGPrefix(ClipSceneREGModel):
    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipSceneREGPrefix, self).train(mode)
        self.gpt.eval()
        return self


def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, "w") as outfile:
        json.dump(config, outfile)


def load_model(
    config_path: str, use_context=True, use_scene=False, epoch_or_latest: Union[str, int] = "_latest"
):
    with open(config_path) as f:
        config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    args = parser.parse_args()
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(args.out_dir, f"{args.prefix}{epoch_or_latest}.pt")
    
    if use_context and not use_scene:  # target + context, loc
        if args.only_prefix:
            model = ClipREGPrefix(args.prefix_length)
        else:
            model = ClipREGModel(args.prefix_length)
    
    elif not use_context and use_scene:  # target + scene + loc:
        if args.only_prefix:
            model = ClipSceneREGPrefix(args.prefix_length)
        else:
            model = ClipSceneREGModel(args.prefix_length)
    
    elif not use_context and not use_scene:  # target + loc
        if args.only_prefix:
            model = ClipNoContextREGPrefix(args.prefix_length)
        else:
            model = ClipNoContextREGModel(args.prefix_length)
            
    else:
        raise NotImplementedError()
    
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    else:
        print(f"{model_path} is not exist")
    return model, parser
