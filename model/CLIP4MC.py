from __future__ import annotations
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import torch
import torch.nn as nn
import numpy as np
from typing import *
from pathlib import Path
from torch.nn import functional as F


from module import build_GPT, build_ViT, build_logit_scale, build_adapter, build_sequence_encoder

from module import CrossEn_Swap, AllGather, VisionTransformer, GPT, AdapterHead, SequenceTransformer

import nltk
from nltk import word_tokenize, pos_tag
from transformers import AutoTokenizer

## Download resources if not already present
nltk.download('all', download_dir="./nltk_data", quiet=True)
nltk.data.path.append(os.path.join(os.getcwd(), Path("./nltk_data")))

tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

allgather = AllGather.apply


def select_embedding(embedding, mask, seq_len):
    ans_embedding = []
    ans_embedding_mask = []
    for msk, emb in zip(mask, embedding):
        select_emb = torch.masked_select(emb, msk.unsqueeze(-1)).view(-1, emb.size(-1))
        tmp_emb = torch.zeros(seq_len, emb.size(-1), device=emb.device)
        tmp_msk = torch.zeros(seq_len, device=emb.device, dtype=torch.bool)
        select_emb = select_emb[:seq_len]
        tmp_emb[:select_emb.size(0)] = select_emb
        tmp_msk[:select_emb.size(0)] = 1
        ans_embedding.append(tmp_emb)
        ans_embedding_mask.append(tmp_msk)
    return torch.stack(ans_embedding, dim=0), torch.stack(ans_embedding_mask, dim=0)


class CLIP4MC(nn.Module):
    def __init__(self,
                 frame_num: int,
                 use_brief_text: bool,
                 use_action: bool,
                 pretrained_clip=None):
        """
        Args:
            frame_num:  number of frames in a video clip
            pretrained_clip: pretrained clip model
        """

        super().__init__()
        self.vit = build_ViT(pretrained_clip)
        self.gpt = build_GPT(pretrained_clip)
        self.sigmoid = torch.nn.Sigmoid()
        self.text_flow = [[self.gpt]]
        self.video_flow = [[self.vit]]

        self.temporal_encoder = build_sequence_encoder('temporal_encoder_config')

        self.video_flow.append([self.temporal_encoder])

        self.video_adapter = build_adapter('video_adapter_config')
        self.text_adapter = build_adapter('text_adapter_config')
        self.video_flow.append([self.video_adapter])
        self.text_flow.append([self.text_adapter])


        self.logit_scale = build_logit_scale()

        self.video_layer_num = [max([module.layers for module in modules]) for modules in self.video_flow]
        self.text_layer_num = [max([module.layers for module in modules]) for modules in self.text_flow]

        self.video_layers = sum(self.video_layer_num)
        self.text_layers = sum(self.text_layer_num)
        self.cross_layers = 1
        self.layers = self.cross_layers + max(self.video_layers, self.text_layers)

        self.frame_num = frame_num
        self.use_action = use_action
        self.use_brief_text = use_brief_text

        self.loss_fct = CrossEn_Swap()

    def get_layer(self, layer: int, layer_type: Literal['video', 'text', 'cross']):
        if layer_type == 'video':
            for i, l in enumerate(self.video_layer_num):
                if layer < l:
                    ans = []
                    for module in self.video_flow[i]:
                        if layer < module.layers:
                            ans += module.get_layer(layer)
                    return ans
                layer -= l
        elif layer_type == 'text':
            for i, l in enumerate(self.text_layer_num):
                if layer < l:
                    ans = []
                    for module in self.text_flow[i]:
                        if layer < module.layers:
                            ans += module.get_layer(layer)
                    return ans
                layer -= l
        elif layer_type == 'cross':
            if layer == 0:
                return self.logit_scale,
        return []

    def get_image_embedding(self, image):
        return self.vit(image)

    def get_video_embedding(self, frame_embedding, motion_frame_embedding=None):

        B, T, D = frame_embedding.shape
        video_embedding = self.temporal_encoder(frame_embedding)
        video_embedding = self.video_adapter(video_embedding)  # (batch, embed_dim)
        video_embedding = video_embedding / video_embedding.norm(dim=-1, keepdim=True)

        return video_embedding


    def get_text_embedding(self, text, entity_mask, action_mask):
        action_mask = action_mask.bool()
        entity_mask = entity_mask.bool()
        text_mask = entity_mask | action_mask

        text_embedding = self.gpt.get_hidden_state(text, full=True)

        text_embedding = text_embedding[torch.arange(text_embedding.shape[0]), text.argmax(dim=-1)]

        text_embedding = self.text_adapter(text_embedding)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        return text_embedding


    def get_logits(self, video_features, text_features):
        video_embedding = video_features


        text_embedding = text_features
        logit1 = self.logit_scale.exp() * video_embedding @ text_embedding.t()
            
        return logit1


    def forward(self, text, entity_mask, action_mask, size, video, motion_input=None, train=False, all_gather=True):
        # video: (batch, frames, channels, height, width)
        # text: (batch, tokens)
        B, T, C, H, W = video.shape
        
        frame_embedding = self.get_image_embedding(video)  # (batch, frames, embed_dim)

        video_embedding = self.get_video_embedding(frame_embedding)

        text_embedding = self.get_text_embedding(text, text, text)

        if all_gather:
            video_embedding = allgather(video_embedding)

            text_embedding = allgather(text_embedding)


        if train:
            size_weight = 1.0
            size_thred = 0.02
            size_sum = 0
            weight_sum = 0
            size_final, size_idx = torch.max(size[:,:], -1)

            gamma_vals = []
            for i in range(B):
                if size_final[i] > size_thred:
                    gamma_val = 1
                else:
                    gamma_val = 0.5 + size_final[i] * 0.5 / size_thred
                gamma_vals.append(gamma_val)

            v2t_matrix = video_embedding @ text_embedding.t()
            v2t_matrix = self.logit_scale.exp() * v2t_matrix
            t2v_matrix = v2t_matrix.t()

            loss = (self.loss_fct(v2t_matrix, gamma_vals) + self.loss_fct(t2v_matrix, gamma_vals)) / 2

            return loss
        else:
            video_features = [self.logit_scale.exp()*video_embedding]

            text_features = text_embedding 
            
            return video_features, text_features

    @torch.no_grad()
    def clamp_logit_scale(self, value=100):
        """
        Follow OpenAI CLIP paper's trick to prevent training instability (sec 2.5)
        """
        self.logit_scale.data.clamp_(-np.log(value), np.log(value))

class CLIP4MC_Restricted(nn.Module):
    def __init__(self,
                 frame_num: int,
                 use_brief_text: bool,
                 use_action: bool,
                 pretrained_clip=None):
        """
        Args:
            frame_num:  number of frames in a video clip
            pretrained_clip: pretrained clip model
        """

        super().__init__()
        self.vit = build_ViT(pretrained_clip)
        self.gpt = build_GPT(pretrained_clip)
        self.sigmoid = torch.nn.Sigmoid()
        self.text_flow = [[self.gpt]]
        self.video_flow = [[self.vit]]

        self.temporal_encoder = build_sequence_encoder('temporal_encoder_config')

        self.video_flow.append([self.temporal_encoder])

        self.video_adapter = build_adapter('video_adapter_config')
        self.text_adapter = build_adapter('text_adapter_config')
        self.video_flow.append([self.video_adapter])
        self.text_flow.append([self.text_adapter])


        self.logit_scale = build_logit_scale()

        self.video_layer_num = [max([module.layers for module in modules]) for modules in self.video_flow]
        self.text_layer_num = [max([module.layers for module in modules]) for modules in self.text_flow]

        self.video_layers = sum(self.video_layer_num)
        self.text_layers = sum(self.text_layer_num)
        self.cross_layers = 1
        self.layers = self.cross_layers + max(self.video_layers, self.text_layers)

        self.frame_num = frame_num
        self.use_action = use_action
        self.use_brief_text = use_brief_text

        self.loss_fct = CrossEn_Swap()

    def get_layer(self, layer: int, layer_type: Literal['video', 'text', 'cross']):
        if layer_type == 'video':
            for i, l in enumerate(self.video_layer_num):
                if layer < l:
                    ans = []
                    for module in self.video_flow[i]:
                        if layer < module.layers:
                            ans += module.get_layer(layer)
                    return ans
                layer -= l
        elif layer_type == 'text':
            for i, l in enumerate(self.text_layer_num):
                if layer < l:
                    ans = []
                    for module in self.text_flow[i]:
                        if layer < module.layers:
                            ans += module.get_layer(layer)
                    return ans
                layer -= l
        elif layer_type == 'cross':
            if layer == 0:
                return self.logit_scale,
        return []

    def get_image_embedding(self, image):
        return self.vit(image)

    def get_video_embedding(self, frame_embedding, motion_frame_embedding=None):

        B, T, D = frame_embedding.shape
        video_embedding = self.temporal_encoder(frame_embedding)
        video_embedding = self.video_adapter(video_embedding)  # (batch, embed_dim)
        video_embedding = video_embedding / video_embedding.norm(dim=-1, keepdim=True)

        return video_embedding


    def get_text_embedding(self, text, entity_mask, action_mask):
        action_mask = action_mask.bool()
        entity_mask = entity_mask.bool()
        text_mask = entity_mask | action_mask

        text_embedding = self.gpt.get_hidden_state(text, full=True)

        text_embedding = text_embedding[torch.arange(text_embedding.shape[0]), text.argmax(dim=-1)]

        text_embedding = self.text_adapter(text_embedding)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        return text_embedding


    def get_logits(self, video_features, text_features):
        video_embedding = video_features


        text_embedding = text_features
        logit1 = self.logit_scale.exp() * video_embedding @ text_embedding.t()
            
        return logit1

    def pos_tagger(self, text):
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        return sum(1 for word, tag in tagged if tag.startswith('NN') or tag.startswith('VB'))        

    def forward(self, text, entity_mask, action_mask, size, video, motion_input=None, train=False, all_gather=True):
        # video: (batch, frames, channels, height, width)
        # text: (batch, tokens)
        B, T, C, H, W = video.shape

        prompts = [tokenizer.decode(t, skip_special_tokens=True) for t in text]
        
        # Embeddings
        frame_embedding = self.get_image_embedding(video)  # (batch, frames, embed_dim)
        video_embedding = self.get_video_embedding(frame_embedding)
        text_embedding = self.get_text_embedding(text, text, text)  # (batch, embed_dim)

        if all_gather:
            video_embedding = allgather(video_embedding)
            text_embedding = allgather(text_embedding)

        if not train:
            # Inference: just return scaled features
            video_features = [self.logit_scale.exp() * video_embedding]
            text_features = text_embedding
            return video_features, text_features

        # --- Training: compute dynamic gamma_vals ---
        # 1. Size-based weight
        size_thred = 0.02
        size_final, _ = torch.max(size, dim=-1)  # (B,)
        gamma_size = torch.zeros(B, device=video_embedding.device)
        for i in range(B):
            if size_final[i] > size_thred:
                gamma_size[i] = 1.0
            else:
                gamma_size[i] = 0.5 + (size_final[i] * 0.5 / size_thred)

        # 2. Relevance: cosine similarity normed to [0,1]
        sim_scores = F.cosine_similarity(video_embedding, text_embedding, dim=-1)  # (B,)
        sim_min, sim_max = sim_scores.min(), sim_scores.max()
        sim_norm = (sim_scores - sim_min) / (sim_max - sim_min + 1e-8)

        # 3. Prompt complexity: count nouns+verbs via POS tagger
        comp_counts = torch.zeros(B, device=video_embedding.device, dtype=torch.float)
        for i, prompt in enumerate(prompts):
            comp_counts[i] = self.pos_tagger(prompt)
        comp_thred = comp_counts.max()
        comp_factor = 0.5 + (comp_counts * 0.5 / (comp_thred + 1e-8))

        # 4. Final gamma values
        gamma_vals = (gamma_size * sim_norm * comp_factor).tolist()

        # --- Contrastive similarity and loss ---
        v2t_matrix = video_embedding @ text_embedding.t()
        v2t_matrix = self.logit_scale.exp() * v2t_matrix
        t2v_matrix = v2t_matrix.t()

        loss_v2t = self.loss_fct(v2t_matrix, gamma_vals)
        loss_t2v = self.loss_fct(t2v_matrix, gamma_vals)
        loss = 0.5 * (loss_v2t + loss_t2v)

        return loss


    @torch.no_grad()
    def clamp_logit_scale(self, value=100):
        """
        Follow OpenAI CLIP paper's trick to prevent training instability (sec 2.5)
        """
        self.logit_scale.data.clamp_(-np.log(value), np.log(value))