# Slm
**Models**
https://drive.google.com/file/d/1H2v7UraBn5CV5lyNFBhHNJoJYpZMHEY_/view?usp=sharing

---

## **1. Model Files**

- The package contains **ONNX format models** for English ↔ Hindi translation:
  - `opus_mt_en_hi_quant.onnx` → English to Hindi
  - `opus_mt_hi_en_quant.onnx` → Hindi to English
- These models are **quantized** (~105 MB each) without affecting translation quality.


## **2. Tokenizer**

Each model has an associated **tokenizer folder** containing:

- `source.spm` and `target.spm` (SentencePiece models)  
- `vocab.json`  
- `special_tokens_map.json`  
- `tokenizer_config.json`  

The **tokenizer must be loaded alongside the model**. It is used to:

1. Convert input text to **input IDs** for the model.  
2. Convert output IDs back to **text**.


## **3. Loading the Model**

- Use **ONNX Runtime for Android** (`onnxruntime-android` or `onnxruntime-android-gpu`) to load the ONNX model.
- Load the corresponding **tokenizer files** to handle input and output processing.


## **4. Handling Addresses and Proper Nouns**

- Some text such as addresses, proper nouns, or uncommon words may not translate correctly.
- Use a **transliteration library** ( `indic-transliteration`) as a fallback.
- The transliteration library should be applied **after translation** for any words that remain in Latin script or are not translated properly.


- Use ONNX Runtime for efficient inference.  
- Integrate the transliteration library to handle special cases where translation alone may not be sufficient.
