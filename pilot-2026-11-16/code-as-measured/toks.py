"""Токенизаторы роутера.

bytes - учебный: один токен равен одному байту UTF-8, id 256 означает конец ответа.
        Кириллица занимает два байта, поэтому буква режется между токенами,
        и тесты ловят битые символы.
hf    - настоящий токенизатор модели через transformers. Версии transformers,
        tokenizers и ревизия модели должны совпадать с репликами.
"""

ASSISTANT_MARK = "<|assistant|>\n"


class ByteTokenizer:
    EOS = 256

    def __init__(self):
        self.stop_ids = [self.EOS]

    def encode_chat(self, messages):
        text = "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in messages)
        return list((text + ASSISTANT_MARK).encode("utf-8"))

    def decode(self, ids):
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


class HFTokenizer:
    def __init__(self, model):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model)
        stop = set()
        if self.tok.eos_token_id is not None:
            stop.add(self.tok.eos_token_id)
        for name in ("<|im_end|>", "<|eot_id|>", "<|endoftext|>"):
            tid = self.tok.convert_tokens_to_ids(name)
            if isinstance(tid, int) and tid >= 0 and tid != self.tok.unk_token_id:
                stop.add(tid)
        self.stop_ids = sorted(stop)

    def encode_chat(self, messages):
        out = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        if hasattr(out, "keys"):
            out = out["input_ids"]
        if out and isinstance(out[0], list):
            out = out[0]
        return [int(i) for i in out]

    def decode(self, ids):
        return self.tok.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )


def make_tokenizer(kind, model):
    if kind == "bytes":
        return ByteTokenizer()
    if kind == "hf":
        return HFTokenizer(model)
    raise ValueError(f"неизвестный токенизатор: {kind}")
