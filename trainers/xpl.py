import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY#, TrainerX, TrainerXU
from utils.trainer import TrainerX, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from losses import SupConLoss

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import copy

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    
    design_details = {"trainer": 'XPL', #'IVLP'
                      "vision_depth": 1,
                      "language_depth": 0, "vision_ctx": cfg.TRAINER.COOP.N_CTX,
                      "language_ctx": 0,
                      "xpl_length": cfg.TRAINER.COOP.N_CTX}

    model = clip.build_model(state_dict or model.state_dict(), design_details)
    #model = clip.build_model(state_dict or model.state_dict())

    return model


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class TextEncoder_deep(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        #print(prompts.shape)
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class MultiModalPromptLearner_CM(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.compound_prompts_depth = 1
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            print("random initialization")
            ctx_vectors_0 = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            ctx_vectors_1 = torch.empty(n_ctx//2, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_0, std=0.02)
            nn.init.normal_(ctx_vectors_1, std=0.02)
            prompt_prefix_0 = " ".join(["X"] * n_ctx)
            prompt_prefix_1 = " ".join(["X"] * (n_ctx//2))
        print('XPL design: Cross-Model Multi-modal Semi-Supervised Prompt Learning')
        print(f'Initial context_pri: "{prompt_prefix_0}"')
        print(f'Initial context_aux: "{prompt_prefix_1}"')
        print(f"Number of XPL context words (tokens) for primary path: {n_ctx}")
        print(f"Number of XPL context words (tokens) for auxiliary path: {(n_ctx//2)}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.half()
        self.ctx_0 = nn.Parameter(ctx_vectors_0)
        self.ctx_1 = nn.Parameter(ctx_vectors_1)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts_0 = [prompt_prefix_0 + " " + name + "." for name in classnames]
        prompts_1 = [prompt_prefix_1 + " " + name + "." for name in classnames]

        tokenized_prompts_0 = torch.cat([clip.tokenize(p) for p in prompts_0])  # (n_cls, n_tkn)
        tokenized_prompts_1 = torch.cat([clip.tokenize(p) for p in prompts_1])
        with torch.no_grad():
            embedding_0 = clip_model.token_embedding(tokenized_prompts_0).type(dtype)
            embedding_1 = clip_model.token_embedding(tokenized_prompts_1).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix_0", embedding_0[:, :1, :])  # SOS
        self.register_buffer("token_suffix_0", embedding_0[:, 1 + n_ctx:, :])  # CLS, EOS
        
        self.register_buffer("token_prefix_1", embedding_1[:, :1, :])  # SOS
        self.register_buffer("token_suffix_1", embedding_1[:, 1 + (n_ctx//2):, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts_0 = tokenized_prompts_0  # torch.Tensor
        self.tokenized_prompts_1 = tokenized_prompts_1
        self.name_lens = name_lens   

    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, use_prompt=None):
        
        
        if use_prompt == 0:
            ctx = self.ctx_0
            prefix = self.token_prefix_0
            suffix = self.token_suffix_0
        if use_prompt == 1:
            ctx = self.ctx_1
            prefix = self.token_prefix_1
            suffix = self.token_suffix_1
    
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = self.construct_prompts(ctx, prefix, suffix)
        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        if use_prompt == 0:
            return prompts, self.proj(self.ctx_0), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required
        if use_prompt == 1:
            return prompts, self.proj(self.ctx_1), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model) #TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()



class CustomCLIP_deep_CM(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        self.prompt_learner = MultiModalPromptLearner_CM(cfg, classnames, clip_model)
        self.tokenized_prompts_0 = self.prompt_learner.tokenized_prompts_0
        self.tokenized_prompts_1 = self.prompt_learner.tokenized_prompts_1

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder_deep(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, use_prompt=None):
        #image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        if use_prompt==0:
            prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner(use_prompt=use_prompt)
            tokenized_prompts = self.tokenized_prompts_0
        elif use_prompt==1:
            prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner(use_prompt=use_prompt)
            tokenized_prompts = self.tokenized_prompts_1
        else:
            print("Exiting! Use an appropriate use_prompt value ...")
            exit()
        
        #print("--shared_ctx:",shared_ctx.shape)
        
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text) #self.text_encoder(prompts, tokenized_prompts)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class XPL(TrainerXU):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP_deep_CM(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
    
    def simclr_loss(self, output_weak, output_strong, criterion, labels=None, normalize=True):
        output_weak = torch.unsqueeze(output_weak, dim=1)
        output_strong = torch.unsqueeze(output_strong, dim=1)
        output_new = torch.cat((output_weak, output_strong), dim=1)
        return criterion(output_new, labels)

    def forward_backward(self, batch_x, batch_u):
        image, image_w, image_s, label = self.parse_batch_train(batch_x, batch_u)
        #print("INPUTS:",image.shape, image_w.shape, image_s.shape, label.shape)
        simclr_loss_criterion = SupConLoss(temperature=0.5)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            #inputs = interleave(torch.cat((image, image_w, image_s)), 2*self.cfg.DATALOADER.TRAIN_U.MU+1).to(self.device)
            inputs = torch.cat((image, image_w, image_s)).to(self.device)
            batch_size = image.shape[0]
            #print("INPUTS non-interleave:{} batch_size:{}".format(inputs.shape, batch_size))
            logits_0 = self.model(inputs, use_prompt=0)
            #logits = de_interleave(logits, 2*self.cfg.DATALOADER.TRAIN_U.MU+1)
            logits_x_0 = logits_0[:batch_size]
            logits_u_w_0, logits_u_s_0 = logits_0[batch_size:].chunk(2)

            logits_1 = self.model(inputs, use_prompt=1)
            #logits = de_interleave(logits, 2*self.cfg.DATALOADER.TRAIN_U.MU+1)
            logits_x_1 = logits_1[:batch_size]
            logits_u_w_1, logits_u_s_1 = logits_1[batch_size:].chunk(2)


            pseudo_label_0 = torch.softmax(logits_u_w_0.detach(), dim=-1)
            max_probs_0, targets_u_0 = torch.max(pseudo_label_0, dim=-1)
            mask_0 = max_probs_0.ge(self.cfg.TRAINER.FIXMATCH.CONF_THRE).float()

            pseudo_label_1 = torch.softmax(logits_u_w_1.detach(), dim=-1)
            max_probs_1, targets_u_1 = torch.max(pseudo_label_1, dim=-1)
            mask_1 = max_probs_1.ge(self.cfg.TRAINER.FIXMATCH.CONF_THRE).float()


            Lu = (F.cross_entropy(logits_u_s_0, targets_u_1) * mask_1).mean() + (F.cross_entropy(logits_u_s_1, targets_u_0) * mask_0).mean()

            Lx = F.cross_entropy(logits_x_0, label) + F.cross_entropy(logits_x_1, label)
            
            
            loss = Lx + self.cfg.TRAINER.FIXMATCH.WEIGHT_U * Lu
            #print("Losses:",Lx.item(), Lu.item())
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(logits_x_0, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        input = batch_x["img"]
        label = batch_x["label"]
        input_w, input_s = batch_u["img"]
        input = input.to(self.device)
        label = label.to(self.device)
        input_w = input_w.to(self.device)
        input_s = input_s.to(self.device)
        return input, input_w, input_s, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
    
    def model_inference(self, input):
        return self.model(input, use_prompt=0)
