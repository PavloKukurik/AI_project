"""
This script sets up a Telegram bot that uses a BART model for text generation and translation.
Modules:
    telebot: A Python library for the Telegram Bot API.
    re: A module for regular expressions.
    os: A module for interacting with the operating system.
    dotenv: A module for loading environment variables from a .env file.
    transformers: A module for state-of-the-art Natural Language Processing for Pytorch and TensorFlow 2.0.
Functions:
    modify_match(match):
        Modifies a matched string by translating it using the BART model.
        Args:
            match (re.Match): The matched object from a regular expression search.
        Returns:
            str: The translated string.
    modify_all_text(text):
        Translates the entire input text using the BART model.
        Args:
            text (str): The input text to be translated.
        Returns:
            str: The translated text.
    query_text(inline_query):
        Handles inline queries from the user, processes the input text, and provides a translation suggestion.
        Args:
            inline_query (telebot.types.InlineQuery): The inline query object from the Telegram bot.
    translate(sentence, **argv):
        Translates a given sentence using the BART model with specified arguments.
        Args:
            sentence (str): The input sentence to be translated.
            **argv: Additional arguments for the BART model's generate method.
        Returns:
            str: The translated sentence.
"""

import telebot
from telebot import types
import re
import os
from dotenv import load_dotenv
from tokenizers import Tokenizer
from transformers import BartTokenizer, BartForConditionalGeneration, AutoModelForCausalLM

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")

bot = telebot.TeleBot(BOT_TOKEN)

save_directory = "./model"
device = "cpu"
model = AutoModelForCausalLM.from_pretrained(save_directory).to(device)
tokenizer_input = BartTokenizer.from_pretrained('facebook/bart-large')
tokenizer_output = Tokenizer.from_file("./tokenizers/hf_tok_emoji.json")

# path = "KomeijiForce/bart-large-emojilm"
# tokenizer = BartTokenizer.from_pretrained(path)
# generator = BartForConditionalGeneration.from_pretrained(path)


def modify_match(match):
    matched_str = match.group(0)
    matched_str = matched_str[1:-1]
    decoded = translate_tuned(matched_str)
    return decoded


def modify_all_text(text):
    return translate_tuned(text)


@bot.inline_handler(lambda query: len(query.query) > 0)
def query_text(inline_query):
    user_input = inline_query.query
    results = re.sub(r"<.*?>", modify_match, user_input)
    suggestions = []
    if results[0] == user_input:
        results = modify_all_text(user_input)
    for inx, result in enumerate(results):
        suggestion_1 = types.InlineQueryResultArticle(
            id=inx,
            title=f"Переклад: {result}",
            input_message_content=types.InputTextMessageContent(f"{result}"),
        )
        suggestions.append(suggestion_1)
    try:
        bot.answer_inline_query(inline_query.id, suggestions)
    except telebot.apihelper.ApiTelegramException as err:
        print(err)
        

# def translate(sentence, **argv):
#     inputs = tokenizer(sentence, return_tensors="pt")
#     generated_ids = generator.generate(inputs["input_ids"], **argv)
#     decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
#     return decoded

def translate_tuned(input_text):
    inputs = tokenizer_input(input_text, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs.input_ids.to(device),
        max_length=50,
        temperature=0.7,
        do_sample=True,
        num_return_sequences=3
    )
    outputs = outputs.cpu()
    lst = []
    for i in range(3):
        output_ids = outputs[i].tolist()
        generated_text = tokenizer_output.decode(output_ids, skip_special_tokens=True)
        generated_text = "".join([text.strip() for text in generated_text.split(" ")])
        lst.append(generated_text)
    print(lst)
    return lst

if __name__ == "__main__":
    bot.infinity_polling()
