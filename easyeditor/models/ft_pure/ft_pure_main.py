from copy import deepcopy
from typing import Any, Dict, List, Tuple
from collections import deque

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util import nethook

from .ft_pure_hparams import FTPureHyperParams
from .generate import generate_fast, set_generate_seed

import time


def apply_ft_pure_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTPureHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas, loss_history = execute_ft_pure(model, tok, requests, hparams)

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            w = nethook.get_parameter(model, w_name)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy, loss_history

import json
import random
def get_augmented_data(hparams, requests, model, tok):
    
    paraphrase_requests = []
    MAX_PARAPHRASE_NUM = hparams.max_paraphrase_num # 10
    MAX_NEIGHBORHOOD_NUM = hparams.max_neighborhood_num #15

    if hparams.generated_prepended_words_path is not None:
        prepended_words = json.load(open(hparams.generated_prepended_words_path, 'r'))["text"]
    else:
        prepended_words = None
        
    if hparams.prompt_neighborhood_path is not None:
        neiborghood_requests = json.load(open(hparams.prompt_neighborhood_path, 'r'))
    else:
        neiborghood_requests = None
        
    
    

    for r in requests:
        # first append the original edit sample 
        augmented_r = [r]

        if hparams.prompt_paraphrase:
            if hparams.prompt_paraphrase_type == "generated_prepended_examples":
                if prepended_words is not None:
                    
                    if hparams.prompt_paraphrase_sample == "fix":
                        sampled_prepended_words = prepended_words[:MAX_PARAPHRASE_NUM]
                    elif hparams.prompt_paraphrase_sample == "random":
                        print("==============Use Random Paraphrase==============")
                        sampled_prepended_words = random.sample(prepended_words, MAX_PARAPHRASE_NUM)
                    paraphrase_prompts = [x + r["prompt"] for x in sampled_prepended_words]
                    
                else:
                    paraphrase_prompts = [x + ". " + r["prompt"] for length, n_gen in hparams.paraphrase_length_params for x in generate_fast(model, tok, ["<|endoftext|>"], n_gen_per_prompt=n_gen,  max_out_len=length)]
                
                for pp in paraphrase_prompts:
                    augmented_r.append({"prompt":pp, "target_new": r["target_new"]})

            else:
                raise ValueError(f"prompt_paraphrase_type {hparams.prompt_paraphrase_type} is not support yet!")

        if hparams.prompt_neighborhood:
            if hparams.prompt_neighborhood_type == "random_examples":
                # List[Dict]
                random.shuffle(neiborghood_requests)
                random_requests_sample = neiborghood_requests[:MAX_NEIGHBORHOOD_NUM]

                for rr in random_requests_sample:
                    augmented_r.append({"prompt": rr["requested_rewrite"]["prompt"].format(rr["requested_rewrite"]["subject"]), "target_new": " " + rr["requested_rewrite"]["target_true"]["str"]})
                    
            elif hparams.prompt_neighborhood_type == "similar_subject":   
                
                similar_subject_sample = neiborghood_requests[r["prompt"]][:MAX_NEIGHBORHOOD_NUM]

                for rr in similar_subject_sample:
                    augmented_r.append({"prompt": rr["prompt"], "target_new": " " + rr["target"]})
                    # Example           
                    # {
                    # "case_id": 0,
                    # "pararel_idx": 2796,
                    # "requested_rewrite": {
                    #     "prompt": "The mother tongue of {} is",
                    #     "relation_id": "P103",
                    #     "target_new": {
                    #         "str": "English",
                    #         "id": "Q1860"
                    #     },
                    #     "target_true": {
                    #         "str": "French",
                    #         "id": "Q150"
                    #     },
                    #     "subject": "Danielle Darrieux"

            else:
                raise ValueError(f"prompt_neighborhood_type {hparams.prompt_neighborhood_type} is not support yet!")
            
    return augmented_r



def execute_ft_pure(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTPureHyperParams,
    **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    device = torch.device(f'cuda:{hparams.device}')
    # model = model.to(device)
    # Update target and print info
    requests = deepcopy(requests)
    for request in requests:
        if request["target_new"] != " ":
            # Space required for correct tokenization
            request["target_new"] = " " + request["target_new"]
        print(
            f"Executing FT Pure algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )
    
    aug_requests =  get_augmented_data(hparams, requests, model, tok)
    assert len(aug_requests) % len(requests) == 0, "Unfinished augmentation for requests!"
    scale_factor = len(aug_requests) // len(requests)
    
    print(f"Augmented Requests Number: {len(aug_requests)}, Scale Factor: {scale_factor}" )
     
    
    # random select layers
    if hparams.random_select_layer:
        print(f"Random select layer from : {hparams.layer_pool}")
        rng = random.Random(int(time.time()))
        hparams.layers = rng.sample(hparams.layer_pool, 1)
        print(f"Select layer: {hparams.layers}")
    
    # Retrieve weights that user desires to change
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n and ("lora_A" in n or "lora_B" in n)
    }

    print("====================Edited Keys=====================")
    print(weights.keys())
    print("====================Edited Keys=====================")
    
    # breakpoint()    
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    print(f"Weights to be updated: {list(weights.keys())}")



    # Define inputs
    texts = [r["prompt"] for r in aug_requests]
    targets = [r["target_new"] for r in aug_requests]
    
    # Configure optimizer / gradients
    opt = torch.optim.Adam(
        [v for _, v in weights.items()],
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    
    #! 添加scheduler
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hparams.num_steps)  # T_max 设置为 100 个 epoch
    
    
    for name, w in model.named_parameters():
        w.requires_grad = name in weights

    # Update loop: intervene at layers simultaneously
    loss_meter = AverageMeter()
    loss_history = []
    for it in range(hparams.num_steps):
        print(20 * "=")
        print(f"Epoch: {it}")
        print(20 * "=")
        loss_meter.reset()

        for txt, tgt in zip(
            chunks(texts, hparams.batch_size*scale_factor), chunks(targets, hparams.batch_size*scale_factor)
        ):
            inputs = tok(txt, return_tensors="pt", padding=True).to(device)
            target_ids = tok(tgt, return_tensors="pt", padding=True)["input_ids"].to(
                device
            )
            if hparams.objective_optimization == 'prompt_last':
                last_token_inds = inputs["attention_mask"].sum(dim=1) - 1
                if tok.unk_token_id is not None:
                    loss_mask = torch.ne(target_ids, tok.unk_token_id)
                else:
                    loss_mask = torch.ones_like(target_ids, dtype=torch.bool)
            elif hparams.objective_optimization == 'target_new':
                inputs_targets = [txt_ + tgt_ for txt_, tgt_ in zip(txt, tgt)]
                inputs_targets = tok(inputs_targets, return_tensors="pt", padding=True).to(device)
                num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in inputs['input_ids'].cpu()]
                num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in inputs_targets['input_ids'].cpu()]
                prompt_len = [x + y for x, y in zip(num_pad_toks, num_prompt_toks)]
                prompt_target_len = inputs_targets['input_ids'].size(1)
                label_mask = torch.tensor([[False] * length + [True] * (prompt_target_len - length) for length in prompt_len]).to(device)
            else:
                print(f"{hparams.objective_optimization} has not been supported yet.")
                raise NotImplementedError
            # last_token_inds = inputs["attention_mask"].sum(dim=1) - 1
            # loss_mask = inputs != tok.unk_token_id
            # loss_mask = [:, ]
            opt.zero_grad()
            bs = inputs["input_ids"].shape[0]
            if 't5' in hparams.model_name.lower():
                inputs['decoder_input_ids'] = target_ids
                logits = model(**inputs).logits
                unmasked_log_probs = logits.log_softmax(-1).gather(-1, inputs['decoder_input_ids'].unsqueeze(-1)).squeeze(-1)

                mask = inputs['decoder_input_ids'] != -100
                n_tokens = mask.float().sum()
                avg_log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
                nll = -avg_log_prob
                loss = nll
            elif 'chatglm' in hparams.model_name.lower():
                # def get_masks(seq, bos_token_id):
                #     """  code from model_chatglm.py  """
                #     if seq.count(bos_token_id) == 2:
                #         context_length = seq[2:].index(bos_token_id) + 2
                #     else:
                #         context_length = seq.index(bos_token_id)
                #     attention_mask = torch.ones((1, len(seq), len(seq)))
                #     attention_mask.tril_()
                #     attention_mask[..., :context_length] = 1
                #     # attention_mask.unsqueeze_(1)
                #     attention_mask = (attention_mask < 0.5).bool()
                #     return attention_mask

                input_ids = inputs['input_ids'].tolist()
                labels = target_ids.tolist()
                assert len(input_ids) == len(labels)
                len_batches = [len(input_ids[i]) + len(labels[i]) + 1
                                 for i in range(len(input_ids))]
                len_max_batch = max(len_batches)
                batch_input_ids = []
                batch_attention_mask = []
                batch_labels = []
                for x, y in zip(input_ids, labels):
                    len_padding = len_max_batch - len(x) - len(y)
                    if tok.padding_side and tok.padding_side == "left":
                        batch_label = [-100] * len_padding + [-100] * len(x) + y
                        batch_input_id = [0] * (len_padding) + x + y
                    else:
                        batch_label = [-100] * len(x) + y + [-100] * len_padding
                        batch_input_id = x + y + [0] * (len_padding)

                    # tensor_attention_mask = get_masks(batch_input_id, bos_token_id=64792)
                    tensor_input_ids = torch.tensor(batch_input_id, dtype=torch.long)
                    tensor_labels = torch.tensor(batch_label, dtype=torch.long)
                    batch_input_ids.append(tensor_input_ids)
                    # batch_attention_mask.append(tensor_attention_mask)
                    batch_labels.append(tensor_labels)
                # batch_attention_mask = torch.stack(batch_attention_mask).to(device)
                batch_input_ids = torch.stack(batch_input_ids).to(device)
                batch_labels = torch.stack(batch_labels).to(device)
                # loss = model(input_ids=batch_input_ids, labels=batch_labels).loss
                lm_logits = model(input_ids=batch_input_ids)['logits']
                lm_logits = lm_logits.to(torch.float32)
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = batch_labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss.to(lm_logits.dtype)
            else:
                if hparams.objective_optimization == 'prompt_last':
                    probs = torch.nn.functional.log_softmax(
                        model(**inputs).logits[torch.arange(bs), last_token_inds], dim=-1
                    )
                    loss = -(torch.gather(probs, 1, target_ids) * loss_mask).sum(
                        1
                    ) / loss_mask.sum(1)
                    loss = loss.mean()
                elif hparams.objective_optimization == 'target_new':
                    logits = model(**inputs_targets).logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = inputs_targets['input_ids'][..., 1:].contiguous()
                    loss_fct = CrossEntropyLoss(reduction='none')
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss = loss.view(bs, -1)
                    loss = (loss * label_mask[:,1:]).sum(1) / label_mask[:,1:].sum(1)
                    loss = loss.mean()
                else:
                    raise NotImplementedError
            print(f"Batch loss {loss.item()}")
            loss_meter.update(loss.item(), n=bs)

            if loss.item() >= hparams.loss_threshold:
                loss.backward()
                opt.step()

            if type(hparams.norm_constraint) is float:
                eps = hparams.norm_constraint
                with torch.no_grad():
                    for k, v in weights.items():
                        v[...] = torch.clamp(
                            v, min=weights_copy[k] - eps, max=weights_copy[k] + eps
                        )

        print(f"Total loss {loss_meter.avg}, lr {opt.param_groups[0]['lr']}")
        # scheduler.step()

        loss_history.append(loss_meter.avg)
        if loss_meter.avg < hparams.loss_threshold:
            break

    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas, loss_history


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
