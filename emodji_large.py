from transformers import BartTokenizer, BartForConditionalGeneration
import sys

def translate(sentence, **argv):
    inputs = tokenizer(sentence, return_tensors="pt")
    generated_ids = generator.generate(inputs["input_ids"], **argv)
    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return decoded

if __name__ == "__main__":
    path = "KomeijiForce/bart-large-emojilm"
    tokenizer = BartTokenizer.from_pretrained(path)
    generator = BartForConditionalGeneration.from_pretrained(path)

    sentence = sys.argv[1]
    decoded = translate(sentence, num_beams=4, do_sample=True, max_length=100)
    print(decoded)
