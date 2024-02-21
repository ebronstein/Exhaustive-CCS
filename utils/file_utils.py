from utils_generation.hf_auth_token import HF_AUTH_TOKEN

FULL_MODEL_NAME_TO_SHORT_NAME = {
    "t5-11b": "t5-11b",
    "google-t5/t5-11b": "google-t5-t5-11b",
    "allenai/unifiedqa-t5-11b": "unifiedqa-t5-11b",
    "bigscience/T0pp": "T0pp",
    "EleutherAI/gpt-j-6B": "gpt-j-6B",
    "roberta-large-mnli": "roberta-large-mnli",
    "microsoft/deberta-v2-xxlarge-mnli": "deberta-v2-xxlarge-mnli",
}


def get_model_short_name(model_str: str, replace_slash: bool = True) -> str:
    """Get the short name of a model.

    Checks the following short names and returns the first one that is valide
    (i.e., it's in the HuggingFace Hub):
    - If the model is in `FULL_MODEL_NAME_TO_SHORT_NAME`, return the corresponding short name.
    - The last part of the model name (e.g., "t5-11b" for "path/to/t5-11b").
    - The last two parts of the model name (e.g., "EleutherAI/gpt-j-6B" for "path/to/EleutherAI/gpt-j-6B").

    If none of the above are valid, return the full model name with "/" replaced by "-".
    """
    if model_str.endswith("/"):
        model_str = model_str[:-1]

    if model_str in FULL_MODEL_NAME_TO_SHORT_NAME:
        return FULL_MODEL_NAME_TO_SHORT_NAME[model_str]

    short_names = [
        model_str.split("/")[-1],
        "/".join(model_str.split("/")[-2:]),
        model_str,
    ]
    for short_name in short_names:
        try:
            AutoConfig.from_pretrained(short_name, use_auth_token=HF_AUTH_TOKEN)
            break
        except OSError:
            pass

    if replace_slash:
        short_name = short_name.replace("/", "-")

    return short_name
