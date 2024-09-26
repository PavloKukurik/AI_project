from transformers import T5Tokenizer, T5ForConditionalGeneration
import sys

path = "KomeijiForce/t5-base-emojilm"
tokenizer = T5Tokenizer.from_pretrained(path)
generator = T5ForConditionalGeneration.from_pretrained(path)

prefix = "translate into emojis:"
sentence = sys.argv[1]
inputs = tokenizer(prefix + " " + sentence, return_tensors="pt")
generated_ids = generator.generate(
    inputs["input_ids"], num_beams=4, do_sample=True, max_length=100
)
decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print(decoded)
