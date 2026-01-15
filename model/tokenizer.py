from transformers import RobertaTokenizer


def build_tokenizer(roberta_tokenizer_path, smiles_maxlen=300):
    return RobertaTokenizer.from_pretrained(
        roberta_tokenizer_path, max_len=smiles_maxlen)
