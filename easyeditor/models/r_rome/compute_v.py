from typing import Dict, List, Tuple

import numpy as np
import torch
from matplotlib.style import context
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome import repr_tools
from ...util import nethook

from .r_rome_hparams import R_ROMEHyperParams


def mcp_loss(v, lambda_mcp=0.01, gamma=3.0):
    """MCP正则化"""
    abs_v = torch.abs(v)
    loss = torch.where(abs_v <= gamma * lambda_mcp,
                       lambda_mcp * abs_v - (v**2 / (2 * gamma)),
                       0.5 * gamma * lambda_mcp**2)
    return torch.sum(loss)

def scad_loss(v, lambda_scad=0.01, a=3.7):
    """SCAD正则化"""
    abs_v = torch.abs(v)
    loss = torch.where(abs_v <= lambda_scad,
                       lambda_scad * abs_v,
                       torch.where(abs_v <= a * lambda_scad,
                                   (-v**2 + 2 * a * lambda_scad * abs_v - lambda_scad**2) / (2 * (a - 1)),
                                   0.5 * (a + 1) * lambda_scad**2))
    return torch.sum(loss)

def compute_v(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: R_ROMEHyperParams,
    layer: int,
    left_vector: torch.Tensor,
    context_templates: List[str],
) -> torch.Tensor:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    target_ids = tok(request["target_new"], return_tensors="pt").to(
        f"cuda:{hparams.device}"
    )["input_ids"][0]

    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]
    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = (
        [
            context.format(request["prompt"]) + tok.decode(target_ids[:-1])
            for context in context_templates
        ],
        ["{} is a"],
    )
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to(f"cuda:{hparams.device}")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device=f"cuda:{hparams.device}").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    # Compute indices of the tokens where the fact is looked up
    vanilla_input_prompts = [
        context.format(request["prompt"]).format(request["subject"])
        for context in context_templates
    ] + [f"{request['subject']} is a"]
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt,
            request["subject"],
            tok,
            hparams.fact_token,
            verbose=(i == 0),
            input_prompt=vanilla_input_prompts[i],
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    if "lora_b" in hparams.mlp_module_tmp.lower():
        if hasattr(model.config, "n_embd"):
            delta = torch.zeros(
                (model.config.n_embd,), requires_grad=True, device=f"cuda:{hparams.device}"
            )
        else:
            delta = torch.zeros(
                (model.config.hidden_size,),
                requires_grad=True,
                device=f"cuda:{hparams.device}",
            )
        
        if "up_proj" in hparams.mlp_module_tmp.lower():
            delta = torch.zeros(
                (11008,),
                requires_grad=True,
                device=f"cuda:{hparams.device}",
            )
        if "gate_proj" in hparams.mlp_module_tmp.lower():
            delta = torch.zeros(
                (11008,),
                requires_grad=True,
                device=f"cuda:{hparams.device}",
            )
        
    elif "lora_a" in hparams.mlp_module_tmp.lower():
        delta = torch.zeros(
            ( hparams.rank,),
            requires_grad=True,
            device=f"cuda:{hparams.device}",
        )
    else:
        raise ValueError(f"lora_a and lora_b not in {hparams.mlp_module_tmp}")
        
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        if cur_layer == hparams.mlp_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0, lookup_idxs[0]].detach().clone()

            noise = torch.rand_like(delta) * min(delta.norm().item(), hparams.delta_noise)
            for i, idx in enumerate(lookup_idxs):
                if len(lookup_idxs) != len(cur_out):
                    cur_out[idx, i, :] += delta + noise 
                    # cur_out[idx, i, :] += delta
                else:
                    cur_out[i, idx, :] += delta + noise 
                    # cur_out[i, idx, :] += delta
                if torch.isnan(cur_out).any() or torch.isinf(cur_out).any():
                    breakpoint()
                    

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits #[22, 23, 32000]

            # Compute distribution for KL divergence
            # breakpoint()
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        max_norm = hparams.clamp_norm_factor * target_init.norm()

        # Compute loss on rewriting targets
        log_probs = torch.log_softmax(logits, dim=2) #[22, 23, 32000]
        #rewriting_targets [21, 23]

        target_probs = torch.gather( log_probs, 2, torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),).squeeze(2)
        mask = (rewriting_targets != -100).float()

        
        #distribution loss
        #L1 loss
        if hparams.dist_loss_type == "l1":
            dist_loss = hparams.dist_loss_factor * torch.norm(delta, p=1)
        elif hparams.dist_loss_type == "mcp":
            dist_loss = mcp_loss(delta, lambda_mcp=hparams.dist_loss_factor, gamma=3.0)
        elif hparams.dist_loss_type == "scad":
            dist_loss = scad_loss(delta, lambda_scad=hparams.dist_loss_factor, a=3.7)
        elif hparams.dist_loss_type == "log":
            dist_loss = hparams.dist_loss_factor * torch.sum(torch.log(1 + torch.abs(delta)))
        elif hparams.dist_loss_type == "l1_log":
            dist_loss = hparams.dist_loss_factor * (torch.norm(delta, p=1) + torch.sum(torch.log(1 + torch.abs(delta))))
        elif hparams.dist_loss_type == "none":
            dist_loss = torch.tensor(0.0).to(delta.device)
        else:
            raise ValueError(f"Incorrect dist_loss_type {hparams.dist_loss_type}")

        # Aggregate total losses
        nll_loss_each = -(target_probs * mask).sum(1) / target_ids.size(0)
        prob_token = (target_probs * mask)[mask==1].reshape(-1, target_ids.size(0)).mean(0)

        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss + weight_decay + dist_loss
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} + {np.round(dist_loss.item(), 3)} "
            f"avg prob of [{request['target_new']}] "
            f"{torch.exp(-nll_loss_each).mean().item()} "
            f"individual prob of [{request['target_new']}] "
            f"{torch.exp(prob_token)}"
        )
        # breakpoint()
        if loss < 5e-2:
            break

        if torch.isnan(loss):
            # breakpoint()
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    if hparams.enable_random_prefix_keys:
        cur_inputs, cur_outputs = [], []
        # run hook for all random prefixes
        for context_template in context_templates:
            cur_input, cur_output = get_module_input_output_at_word(
                model,
                tok,
                layer,
                context_template=context_template.format(request["prompt"]),
                word=request["subject"],
                module_template=hparams.rewrite_module_tmp,
                fact_token_strategy=hparams.fact_token,
            )
            cur_inputs.append(cur_input)
            cur_outputs.append(cur_output)

        # average the representations across prefixes
        cur_input = torch.stack(cur_inputs).mean(0)
        cur_output = torch.stack(cur_outputs).mean(0)

        # target_init is v*, based on output from random prefix computations
        target = target_init + delta.to(target_init.dtype)
    else:
        # Original ROME code
        # Retrieve cur_input, the current input to the 2nd MLP layer, and
        # cur_output, the original output of the 2nd MLP layer.
        cur_input, cur_output = get_module_input_output_at_word(
            model,
            tok,
            layer,
            context_template=request["prompt"],  # only done for the prompt being edited
            word=request["subject"],
            module_template=hparams.rewrite_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )

        # cur_output is v, based on output from prompt-only computations
        target = cur_output + delta.to(target_init.dtype)

    # Solving the linear system to compute the right vector
    # breakpoint()
    right_vector = (target - cur_output) / torch.dot(cur_input, left_vector)
    print(f"target_init - cur_output norm: {(target_init - cur_output).norm()}")
    print(f"Delta norm: {(target - cur_output).norm().item()}")
    print(f"Max norm: {max_norm.item()}")
    print(
        f"Change in target norm: {target_init.norm().item()} to {target.norm().item()} => {(target.norm() - target_init.norm()).item()}"
    )
    print(f"Division Factor: {torch.dot(cur_input, left_vector).item()}")
    print(f"Right vector norm: {right_vector.norm()}")

    # breakpoint()
    # return right_vector
    return right_vector, loss


def get_module_input_output_at_word(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_template: str,
    word: str,
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both",
            subtoken=subtoken,
            context_templates=[context_template],
            words=[word],
            **word_repr_args,
        )
    elif fact_token_strategy == "last":
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both",
            contexts=[context_template.format(word)],
            idxs=[[-1]],
            **word_repr_args,
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    l_input, l_output = l_input[0], l_output[0]
    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
    input_prompt=None,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = len(tok.encode(input_prompt)) - 1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret
