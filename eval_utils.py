
def generate(model, tok, text, max_new_tokens=32, do_smaple=False, answer_only=True):
    # model.eval()

    ori_tok_padding_side = tok.padding_side
    tok.padding_side = "left"
    if model.training:
        raise ValueError("The model should in eval model in inference!")

    text_token = tok(text, padding=True, truncation=True, return_tensors="pt").to(
        model.device
    )

    answer_token = model.generate(
        **text_token, max_new_tokens=max_new_tokens, do_sample=do_smaple
    )

    question_answer_batch = tok.batch_decode(answer_token, skip_special_tokens=True)
    answer_batch = [ans[len(t) :] for t, ans in zip(text, question_answer_batch)]
    
    tok.padding_side = ori_tok_padding_side

    if answer_only:
        return answer_batch
    else:
        return question_answer_batch


def paraphrase(
    rephrase_model,
    rephrase_tokenizer,
    question,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=5,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128,
):
    input_ids = rephrase_tokenizer(
        f"paraphrase: {question}",
        return_tensors="pt",
        padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids.to(rephrase_model.device)

    outputs = rephrase_model.generate(
        input_ids,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        max_length=max_length,
        diversity_penalty=diversity_penalty,
    )

    res = rephrase_tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res


def get_rephrase_text(rephrase_model, rephrase_tok, p, prompt_idx_map, eval_facts, nums):
    rephrase_text = [p]

    if (idx := prompt_idx_map.get(p, None)) is not None:
        rephrase_text_generate = eval_facts[idx]["rephrase"][:nums]
        # return eval_facts[idx]["rephrase"][:nums]
    else:
        print(f"Obtian the rephrase text for {p}")
        rephrase_text_generate = paraphrase(
            rephrase_model, rephrase_tok,
            p, num_return_sequences=nums, num_beams=nums, num_beam_groups=nums
        )
        eval_facts.append({"prompt": p, "rephrase": rephrase_text_generate})
        prompt_idx_map[p] = len(eval_facts) - 1

    rephrase_text.extend(rephrase_text_generate)
    return rephrase_text
