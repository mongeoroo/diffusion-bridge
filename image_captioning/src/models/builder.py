from transformers import GPT2LMHeadModel, GPT2Config


def build_huggingface_model(hf_model_name):
    if hf_model_name == "gpt2":
        config = GPT2Config.from_pretrained(hf_model_name)
        return GPT2LMHeadModel.from_pretrained(hf_model_name, config=config)
    else:
        raise NotImplementedError(
            f"Huggingface model builder not implemented for {hf_model_name}"
        )

