
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from functools import reduce
from operator import mul
from copy import deepcopy
from torch.nn.modules.utils import _pair
from torch.nn.modules.loss import CrossEntropyLoss
from clip_modules.clip_model import load_clip, QuickGELU
from clip_modules.tokenization_clip import SimpleTokenizer
from model.common import *


class COOP_Multi_Path(nn.Module):
    
    def __init__(self, config, attributes, classes, offset):
        super().__init__()
        self.clip = load_clip(name=config.clip_arch, context_length=config.context_length)
        self.tokenizer = SimpleTokenizer()
        self.config = config
        self.attributes = attributes
        self.classes = classes
        self.attr_dropout = nn.Dropout(config.attr_dropout)

        self.token_ids, self.soft_att_obj, comp_ctx_vectors, attr_ctx_vectors, obj_ctx_vectors = self.construct_soft_prompt()
        self.offset = offset
        self.enable_pos_emb = True
        dtype = self.clip.dtype
        if dtype is None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        self.text_encoder = CustomTextEncoder(self.clip, self.tokenizer, self.dtype)

        # only consider ViT as visual encoder
        assert 'ViT' in config.clip_model

        self.soft_att_obj = nn.Parameter(self.soft_att_obj)
        self.comp_ctx_vectors = nn.Parameter(comp_ctx_vectors).cuda()
        self.attr_ctx_vectors = nn.Parameter(attr_ctx_vectors).cuda()
        self.obj_ctx_vectors = nn.Parameter(obj_ctx_vectors).cuda()
        
        # freeze all
        for p in self.parameters():    # freeze current parameters
            p.requires_grad = False

        self.comp_ctx_vectors.requires_grad = True
        self.attr_ctx_vectors.requires_grad = True
        self.obj_ctx_vectors.requires_grad = True


    def encode_image(self, x: torch.Tensor):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        img_feature = self.clip.visual.transformer(x)

        img_feature = img_feature.permute(1, 0, 2)  # LND -> NLD

        img_feature = self.clip.visual.ln_post(img_feature)
        if self.clip.visual.proj is not None:
            img_feature = img_feature @ self.clip.visual.proj
        return img_feature[:, 0, :], img_feature


    def encode_text(self, token_ids, token_tensors=None, enable_pos_emb=False):
        return self.text_encoder(token_ids, token_tensors, enable_pos_emb)


    def construct_soft_prompt(self):
        # token_ids indicates the position of [EOS]
        token_ids = self.tokenizer(self.config.prompt_template,
                              context_length=self.config.context_length).cuda()

        tokenized = torch.cat(
            [
                self.tokenizer(tok, context_length=self.config.context_length)
                for tok in self.attributes + self.classes
            ]
        )
        orig_token_embedding = self.clip.token_embedding(tokenized.cuda())
        soft_att_obj = torch.zeros(
            (len(self.attributes) + len(self.classes), orig_token_embedding.size(-1)),
        )
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            soft_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        ctx_init = self.config.ctx_init
        assert isinstance(ctx_init, list)
        n_ctx = [len(ctx.split()) for ctx in ctx_init]
        prompt = self.tokenizer(ctx_init,
                            context_length=self.config.context_length).cuda()
        with torch.no_grad():
            embedding = self.clip.token_embedding(prompt)

        comp_ctx_vectors = embedding[0, 1 : 1 + n_ctx[0], :].to(self.clip.dtype)
        attr_ctx_vectors = embedding[1, 1 : 1 + n_ctx[1], :].to(self.clip.dtype)
        obj_ctx_vectors = embedding[2, 1 : 1 + n_ctx[2], :].to(self.clip.dtype)
        
        return token_ids, soft_att_obj, comp_ctx_vectors, attr_ctx_vectors, obj_ctx_vectors


    def construct_token_tensors(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        token_tensor, num_elements = list(), [len(pair_idx), self.offset, len(self.classes)]
        for i_element in range(self.token_ids.shape[0]):
            class_token_ids = self.token_ids[i_element].repeat(num_elements[i_element], 1)
            token_tensor.append(self.clip.token_embedding(
                class_token_ids.cuda()
            ).type(self.clip.dtype))

        eos_idx = [int(self.token_ids[i_element].argmax()) for i_element in range(self.token_ids.shape[0])]
        soft_att_obj = self.attr_dropout(self.soft_att_obj)
        # comp
        token_tensor[0][:, eos_idx[0] - 2, :] = soft_att_obj[
            attr_idx
        ].type(self.clip.dtype)
        token_tensor[0][:, eos_idx[0] - 1, :] = soft_att_obj[
            obj_idx + self.offset
        ].type(self.clip.dtype)
        token_tensor[0][
            :, 1 : len(self.comp_ctx_vectors) + 1, :
        ] = self.comp_ctx_vectors.type(self.clip.dtype)
        # attr
        token_tensor[1][:, eos_idx[1] - 1, :] = soft_att_obj[
            :self.offset
        ].type(self.clip.dtype)
        token_tensor[1][
            :, 1 : len(self.attr_ctx_vectors) + 1, :
        ] = self.attr_ctx_vectors.type(self.clip.dtype)
        # obj
        token_tensor[2][:, eos_idx[2] - 1, :] = soft_att_obj[
            self.offset:
        ].type(self.clip.dtype)
        token_tensor[2][
            :, 1 : len(self.obj_ctx_vectors) + 1, :
        ] = self.obj_ctx_vectors.type(self.clip.dtype)

        return token_tensor
    

    def loss_calu(self, predict, target):
        loss_fn = CrossEntropyLoss()
        _, batch_attr, batch_obj, batch_target = target
        comp_logits, attr_logits, obj_logits = predict
        batch_attr = batch_attr.cuda()
        batch_obj = batch_obj.cuda()
        batch_target = batch_target.cuda()
        loss_comp = loss_fn(comp_logits, batch_target)
        loss_attr = loss_fn(attr_logits, batch_attr)
        loss_obj = loss_fn(obj_logits, batch_obj)
        loss = loss_comp * self.config.pair_loss_weight +\
               loss_attr * self.config.attr_loss_weight +\
               loss_obj * self.config.obj_loss_weight
        return loss


    def logit_infer(self, predict, pairs):
        comp_logits, attr_logits, obj_logits = predict
        attr_pred = F.softmax(attr_logits, dim=-1)
        obj_pred = F.softmax(obj_logits, dim=-1)
        for i_comp in range(comp_logits.shape[-1]):
            comp_logits[:, i_comp] = comp_logits[:, i_comp] + \
                                    attr_pred[:, pairs[i_comp][0]] * obj_pred[:, pairs[i_comp][1]]
        return comp_logits


    def forward(self, batch, idx):
        batch_img = batch[0].cuda()
        b = batch_img.shape[0]
        l, _ = idx.shape
        batch_img, batch_patch = self.encode_image(batch_img.type(self.clip.dtype))
        batch_img_features = [batch_img, batch_img, batch_img]
        normalized_img_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in batch_img_features]

        token_tensors = self.construct_token_tensors(idx)

        logits = list()
        for i_element in range(self.token_ids.shape[0]):
            _text_features, _ = self.encode_text(
                self.token_ids[i_element],
                token_tensors[i_element],
                enable_pos_emb=self.enable_pos_emb,
            )

            idx_text_features = _text_features / _text_features.norm(
                dim=-1, keepdim=True
            )

            logits.append(
                torch.einsum(
                    "bd, kd->bk", 
                    normalized_img_features[i_element], 
                    idx_text_features * self.clip.logit_scale.exp()
            ))

        return logits