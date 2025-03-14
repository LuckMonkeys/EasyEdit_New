from copy import deepcopy
from typing import Any, Dict, List, Tuple
from collections import deque

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util import nethook

from .ft_plus_hparams import FTPlusHyperParams
from .generate import generate_fast, set_generate_seed
from peft import set_peft_model_state_dict, get_peft_model_state_dict
import time

def apply_ft_plus_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTPlusHyperParams,
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
        
        
        
    prev_global_model = kwargs.get("prev_global_model", None)
    norm_before_edit = calculate_norm(get_peft_model_state_dict(model), prev_global_model)

    deltas, loss_history = execute_ft_plus(model, tok, requests, hparams, **kwargs)

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            w = nethook.get_parameter(model, w_name)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix

    print(f"New weights successfully inserted into {list(deltas.keys())}")
    
    
    norm_after_edit = calculate_norm(get_peft_model_state_dict(model), prev_global_model)
    
    print(f"Norm Before Edit: {norm_before_edit}, Norm After Edit: {norm_after_edit}")

    return model, weights_copy, loss_history

import json
import random
def get_augmented_data(hparams, requests, model, tok):
    
    paraphrase_requests = []
    MAX_PARAPHRASE_NUM = hparams.max_paraphrase_num # 10
    MAX_NEIGHBORHOOD_NUM = hparams.max_neighborhood_num #15
    MAX_ARCA_NUM = hparams.max_arca_num

    if hparams.generated_prepended_words_path is not None:
        prepended_words = json.load(open(hparams.generated_prepended_words_path, 'r'))["text"]
    else:
        prepended_words = None
        
    if hparams.prompt_neighborhood_path is not None:
        neiborghood_requests = json.load(open(hparams.prompt_neighborhood_path, 'r'))
    else:
        neiborghood_requests = None
        
    if hparams.rephrase_facts_path is not None:
        repharase_facts = json.load(open(hparams.rephrase_facts_path, 'r'))
    else:
        repharase_facts = None
        
    if hparams.arca_prompts_path is not None and hparams.arca_prompts_path != "":
        arac_prompts = json.load(open(hparams.arca_prompts_path, 'r'))
    else:
        arac_prompts = None
    

    total_augmented_r = []
    
    for r in requests:
        # first append the original edit sample 
        augmented_r = [r]
        paraphrase_prompts = []
        
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
                
                
            elif hparams.prompt_paraphrase_type == "rephrase_examples":
                for fact_dict in repharase_facts:
                    if fact_dict["prompt"] == r["prompt"]:
                        
                        if hparams.prompt_paraphrase_sample == "fix":
                            paraphrase_prompts = fact_dict["rephrase"][:MAX_PARAPHRASE_NUM]
                        elif hparams.prompt_paraphrase_sample == "random":
                            print("==============Use Random Paraphrase==============")
                            
                            rng = random.Random(int(time.time()))
                            paraphrase_prompts = rng.sample(fact_dict["rephrase"], MAX_PARAPHRASE_NUM)
        
                if len(paraphrase_prompts) == 0:
                    raise ValueError(f"""No Rephrase Prompt Found for {r["prompt"]}""")
            else:
                raise ValueError(f"prompt_paraphrase_type {hparams.prompt_paraphrase_type} is not support yet!")

            for pp in paraphrase_prompts:
                augmented_r.append({"prompt":pp, "target_new": r["target_new"]})
            
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
            
        if hparams.arca_prompts:
            # arca_format = "The following questions are rephrased versions. Answer them identically: Question 1: {}  Question 2: {}  Answer:"
            # arca_format = "Answer them identically: Question 1: {}  Question 2: {}"
            # arca_format = "{} {}" # 扩展性差
            # arca_format = "The rephrased version of {} is {}"
            
            synonyms_list = [
                "amounts to",
                "is equivalent to",
                "is the same as",
                "is identical to",
                "corresponds to",
                "matches",
                "represents",
                "makes up",
                "balances",
                "is commensurate with"
            ]
            
            arca_format = "{} {} {}"
            
            
            
            
            import time
            total_arca_prompt = []
            # for prompt_tok in arac_prompts[r["prompt"]].values():
            #     total_arca_prompt += prompt_tok
            # rng = random.Random(int(time.time()))
            # sample_arca_prompts = rng.sample(total_arca_prompt, MAX_ARCA_NUM)
            # sample_arca_prompt = rng.sample(total_arca_prompt, 1)[0]
            
            sample_arca_prompt = f"Output {r['target_new']}"
            print("===============Selected sample_arca_prompt===", sample_arca_prompt)

            for synon in synonyms_list[:MAX_ARCA_NUM]:
                augmented_r.append({"prompt": arca_format.format(r["prompt"], synon, sample_arca_prompt), "target_new": sample_arca_prompt})
            
            # for arca in sample_arca_prompts:
                # augmented_r.append({"prompt": arca_format.format(arca, r["prompt"]), "target_new": r["target_new"]})
                # augmented_r.append({"prompt": arca, "target_new": r["target_new"]})
                # augmented_r.append({"prompt": arca_format.format(r["prompt"], arca), "target_new": arca})
                
                

            
        total_augmented_r += augmented_r
        # breakpoint()        
    return total_augmented_r

def get_global_grad_mask(rewrite_weight, trained_model, prev_global_model, ratio):
    key_order = list(rewrite_weight.keys())
    grad_list = []
    
    for p_name in key_order:
        # delete ".default" from p_name
        p_name = p_name.replace(".default", "")
        grad_list.append(trained_model[p_name] - prev_global_model[p_name])
    
    # concatenate all the gradients into one-deimensional tensor
    grad_tensor = torch.cat([ g.abs().view(-1) for g in grad_list])
    
    #get the top-k smallest gradient mask
    grad_mask = torch.zeros_like(grad_tensor)
    grad_mask[torch.topk(grad_tensor, int(grad_tensor.size(0) * ratio), largest=False).indices] = 1
    
    #reshape the mask to the same shape as the original gradient
    grad_mask_dict = {}
    start = 0
    for p_name in key_order:
        end = start + rewrite_weight[p_name].numel()
        grad_mask_dict[p_name] = grad_mask[start:end].view_as(rewrite_weight[p_name])
        start = end
    return grad_mask_dict

def get_layerwise_grad_mask(rewrite_weight, trained_model, prev_global_model, ratio):
    # indepently calculate the gradient mask for each parameter
    grad_mask_dict = {}
    for p_name in rewrite_weight.keys():
        dict_p_name_ = p_name.replace(".default", "")
        grad = trained_model[dict_p_name_] - prev_global_model[dict_p_name_]
        grad_flat = grad.abs().flatten()
        grad_mask = torch.zeros_like(grad_flat)
        grad_mask[torch.topk(grad_flat, int(grad_flat.numel() * ratio), largest=False).indices] = 1
        grad_mask_dict[p_name] = grad_mask.view_as(grad)
    return grad_mask_dict

def calculate_norm(model_dict_a, model_dict_b):
    # breakpoint()
    norm = 0
    for k in model_dict_b.keys():
        norm += torch.norm(model_dict_a[k] - model_dict_b[k], p=2)
    return norm


def execute_ft_plus(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTPlusHyperParams,
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
            f"Executing FT Plus algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )
    
    aug_requests =  get_augmented_data(hparams, requests, model, tok)
    # breakpoint()
    assert len(aug_requests) % len(requests) == 0, "Unfinished augmentation for requests!"
    scale_factor = len(aug_requests) // len(requests)
    
    print(f"Augmented Requests Number: {len(aug_requests)}, Scale Factor: {scale_factor}" )
     
    # Retrieve weights that user desires to change
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n and ("lora_A" in n or "lora_B" in n)
    }
    
    # attn_weight = {
    #     n: p
    #     for n, p in model.named_parameters()
    #     for layer in hparams.layers
    #     if hparams.attn_module_tmp.format(layer) in n and ("lora_A" in n or "lora_B" in n)
    # }
    
    # mlp_weight = {
    #     n: p
    #     for n, p in model.named_parameters()
    #     for layer in hparams.layers
    #     if hparams.mlp_module_tmp.format(layer) in n and ("lora_A" in n or "lora_B" in n)
    # }
    
    
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
    
    #! 记得取消注释
    # for name, w in model.named_parameters():
    #     w.requires_grad = name in weights
    
    # !为了查看所有参数的梯度，将所有参数的梯度都设置为True
    for name, w in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            w.requires_grad = True
    
    # print("================Apply Scheduler======================")
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hparams.num_steps)  # T_max 设置为 100 个 epoch
    

    # Update loop: intervene at layers simultaneously
    loss_meter = AverageMeter()
    loss_history = []
    
    if hparams.apply_grad_mask:
        prev_global_model = kwargs.get("prev_global_model", None)
        # breakpoint()
        assert prev_global_model is not None, "prev_global_model is None"
        
        if hparams.grad_mask_type == 'global':
            grad_mask_dict = get_global_grad_mask(weights, get_peft_model_state_dict(model), prev_global_model , hparams.grad_mask_ratio)
        elif hparams.grad_mask_type == 'layerwise':
            grad_mask_dict = get_layerwise_grad_mask(weights, get_peft_model_state_dict(model), prev_global_model , hparams.grad_mask_ratio)
        else:
            raise ValueError(f"grad_mask_type {hparams.grad_mask_type} is not support yet!")
    
    

    
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
                
            ## constrain the new_weight - old_weight
            norm_loss = 0
            if hparams.l2_norm_constraint > 0:
                for name in weights.keys():
                    # p_name = name.replace(".default", "")
                    norm_loss += hparams.l2_norm_constraint * torch.norm(weights[name] - weights_copy[name], p=2)
            
            print(f"Batch loss: ACC {loss.item()}, Norm {norm_loss}")
            total_loss = loss + norm_loss
            
            loss_meter.update(total_loss.item(), n=bs)

            if total_loss.item() >= hparams.loss_threshold:
                total_loss.backward()
                
                # breakpoint()
                if hparams.layer_grad_magnitude:
                    print("===============Layer Grad Magnitude=============")
                    #calcuate the gradient magnitude for mlp, atten and layer
                    total_layers = model.config.num_hidden_layers
                    grad_layer_list, grad_mlp_list, grad_attn_list = [], [], []
                    for layer in range(total_layers):
                        layer_grad, layer_mlp_grad, layer_attn = 0, 0, 0
                        layer_numl, layer_mlp_numl, layer_attn_numl = 0, 0, 0
                        for p_name, p in model.named_parameters():
                            if f"model.layers.{layer}." in p_name and ("lora_A" in p_name or "lora_B" in p_name):
                                # breakpoint()
                                try:
                                    layer_grad += p.grad.norm()
                                    layer_numl += p.numel()
                                except:
                                    breakpoint()
                                
                                if "mlp" in p_name:
                                    layer_mlp_grad += p.grad.norm()
                                    layer_mlp_numl += p.numel()
                                elif "attn" in p_name:
                                    layer_attn += p.grad.norm()
                                    layer_attn_numl += p.numel()
                                    
                        grad_layer_list.append(layer_grad / layer_numl )
                        grad_mlp_list.append(layer_mlp_grad /layer_mlp_numl )
                        grad_attn_list.append(layer_attn / layer_attn_numl)
                                                        
                        print(f"Layer {layer} Gradient Norm: {layer_grad.item() / layer_numl}, MLP Norm: {layer_mlp_grad.item() / layer_mlp_numl}, Attn Norm: {layer_attn.item() / layer_attn_numl}")
                    
                    # show layer idx with greatest grad
                    layer_idx = grad_layer_list.index(max(grad_layer_list))
                    mlp_idx = grad_mlp_list.index(max(grad_mlp_list))
                    attn_idx = grad_attn_list.index(max(grad_attn_list))
                    print(f"Layer with Greatest Gradient: {layer_idx}, MLP: {mlp_idx}, Attn: {attn_idx}")
                
                
                
                            
                #! Apply grad mask
                if hparams.apply_grad_mask:
                    print("===============Apply Gradient Mask=============")
                    for p_name, p in weights.items():
                        p.grad *= grad_mask_dict[p_name]
                        
                elif hparams.large_grad_only:
                    print(f"===============Apply {hparams.largest_grad_ratio} Large Gradient=============")
                    for p_name, p in weights.items():
                        if p.grad is not None:
                            # Flatten the gradient tensor
                            grad_flat = p.grad.abs().flatten()
                            # Get indices of top k% largest gradients
                            k = int(grad_flat.numel() * hparams.largest_grad_ratio)  # Keep top 10%, adjust as needed
                            _, indices = torch.topk(grad_flat, k)
                            # Create mask of zeros with ones at top k indices
                            mask = torch.zeros_like(grad_flat)
                            mask[indices] = 1
                            # Reshape mask back to gradient shape and apply
                            p.grad *= mask.reshape(p.grad.shape)
                        
 
                
                
                opt.step()

            if type(hparams.norm_constraint) is float:
                eps = hparams.norm_constraint
                with torch.no_grad():
                    for k, v in weights.items():
                        max_diff = torch.max((v - weights_copy[k]).abs())
                        # print(f"Weight {k}, Max Diff {max_diff.item()}")
                        print(f"Before Clamp: {k}, Base Norm: {weights_copy[k].norm().item()}, Edit Norm: {v.norm().item()}, Max Diff {max_diff}")
                        # breakpoint()
                        v[...] = torch.clamp(
                            v, min=weights_copy[k] - eps, max=weights_copy[k] + eps
                        )
                        print(f"After Clamp: {k}, Base Norm: {weights_copy[k].norm().item()}, Edit Norm: {v.norm().item()}")

        print(f"Total loss {loss_meter.avg}")
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
