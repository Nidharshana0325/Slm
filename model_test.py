import onnxruntime as ort
import numpy as np
from transformers import MarianTokenizer
import re
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

class ONNXTranslator:
    def __init__(self, model_path, tokenizer_dir):
        self.session = ort.InferenceSession(model_path)
        self.tokenizer = MarianTokenizer.from_pretrained(tokenizer_dir)

    def translate(self, text, max_len=50):
        # Encode input
        inputs = self.tokenizer(text, return_tensors="np")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        decoder_ids = np.array([[self.tokenizer.pad_token_id]])

        for _ in range(max_len):
            ort_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "decoder_input_ids": decoder_ids
            }
            logits = self.session.run(None, ort_inputs)[0]
            next_token_id = np.argmax(logits[:, -1, :], axis=-1)
            decoder_ids = np.concatenate([decoder_ids, next_token_id[:, None]], axis=-1)
            if next_token_id[0] == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(decoder_ids[0], skip_special_tokens=True)

def is_symbol_heavy(text):
    letters = len(re.findall(r'[A-Za-z0-9]', text))
    total = len(text)
    return total > 0 and (letters / total) < 0.3

def transliterate_chunk(text):
    return re.sub(r"[A-Za-z0-9]+",
                  lambda m: transliterate(m.group(0), sanscript.ITRANS, sanscript.DEVANAGARI),
                  text)

def split_text(text):
    sentences = re.split(r'([।.!?;•])', text)
    chunks = []
    for i in range(0, len(sentences)-1, 2):
        chunks.append(sentences[i].strip() + sentences[i+1].strip())
    if len(sentences) % 2 != 0:
        chunks.append(sentences[-1].strip())
    return [c for c in chunks if c]

def clean_translation(text):
    text = re.sub(r"[⁇]+", "", text)
    text = re.sub(r"([।,.!?])\1+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()

def split_sentences(text):
    sentences = re.split(r'([।.!?])', text)
    result = []
    for i in range(0, len(sentences)-1, 2):
        result.append(sentences[i].strip() + sentences[i+1].strip())
    if len(sentences) % 2 != 0:
        result.append(sentences[-1].strip())
    return [s for s in result if s]

class EN_HI_Translator:
    def __init__(self, model_path, tokenizer_dir):
        self.translator = ONNXTranslator(model_path, tokenizer_dir)

    def translate(self, text):
        chunks = split_text(text)
        translated_chunks = []

        for chunk in chunks:
            if is_symbol_heavy(chunk):
                translated_chunks.append(transliterate_chunk(chunk))
            else:
                translated_chunks.append(clean_translation(self.translator.translate(chunk)))

        return " ".join(translated_chunks)

class HI_EN_Translator:
    def __init__(self, model_path, tokenizer_dir):
        self.translator = ONNXTranslator(model_path, tokenizer_dir)

    def translate(self, text):
        sentences = split_sentences(text)
        translated_sentences = []

        for s in sentences:
            
            translated_sentences.append(clean_translation(self.translator.translate(s)))

        return " ".join(translated_sentences)

class Translator:
    def __init__(self):
        self.en_hi = EN_HI_Translator(
            "/kaggle/working/TranslationPackage/EN_HI/opus_mt_en_hi_quant.onnx",
            "/kaggle/working/TranslationPackage/EN_HI/tokenizer_en_hi"
        )
        self.hi_en = HI_EN_Translator(
            "/kaggle/working/TranslationPackage/HI_EN/opus_mt_hi_en_quant.onnx",
            "/kaggle/working/TranslationPackage/HI_EN/tokenizer_hi_en"
        )

    def translate(self, text, direction="EN->HI"):
        if direction == "EN->HI":
            return self.en_hi.translate(text)
        else:
            return self.hi_en.translate(text)

if __name__ == "__main__":
    translator = Translator()

    text_en = "My name is Rahul and I live in Delhi. 56, Green Valley Residency, Mumbai."
    text_hi = "मेरा नाम राहुल है और मैं दिल्ली में रहता हूँ। 56, ग्रीन वैली रेसिडेंसी, मुंबई।"

    print("EN -> HI:")
    print(translator.translate(text_en, direction="EN->HI"))

    print("\nHI -> EN:")
    print(translator.translate(text_hi, direction="HI->EN"))
