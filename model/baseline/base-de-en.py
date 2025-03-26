from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
import sacrebleu

model_name = "Helsinki-NLP/opus-mt-de-nl"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Load the development (dev) data
dev_df = pd.read_csv("de_en_dev.csv")
source_texts = dev_df["source"].tolist()

def translate_texts(texts, tokenizer, model, batch_size=16, max_new_tokens=128):

    translations = []
    # Process texts in batches to avoid memory issues
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        
        # Generate translations with a cap on the maximum new tokens
        translated_tokens = model.generate(**inputs, max_new_tokens=max_new_tokens)
        batch_translations = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
        translations.extend(batch_translations)
    return translations

# Generate translations using the dev set in manageable batches
translations = translate_texts(source_texts, tokenizer, model, batch_size=16, max_new_tokens=128)

# Save predictions in the dev dataframe for inspection
dev_df["predictions"] = translations
dev_df.to_csv("de_en_dev_predictions.csv", index=False)

# Evaluate using BLEU (SacreBLEU expects a list of reference corpora)
references = [dev_df["target"].tolist()]  # list of one reference corpus
bleu = sacrebleu.corpus_bleu(translations, references)
print("Dev BLEU score:", bleu.score)