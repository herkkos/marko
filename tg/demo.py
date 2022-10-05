#!/usr/bin/env python
import os
import random
import time

from Levenshtein import distance as lev
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from transformers import BertTokenizer

from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext, MessageHandler, Filters

TOKEN = "INSERT HERE"

URL = "INSERT HERE"
IP = "0.0.0.0"
PORT = int(os.environ.get('PORT', 443))

# MP
with open('mp_list.txt', 'r', encoding="utf-8") as f:
    MP = f.readlines()
OLD_MP = {}

def get_time(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /get is issued."""
    timestamp = round(time.time() * 1000)
    update.message.reply_text(str(timestamp))


def get_paja(update: Update, context: CallbackContext) -> None:
    link = "https://paihdelinkki.fi/"
    update.message.reply_text("Täältä saat apua\n" + link)


def get_mp(update: Update, context: CallbackContext) -> None:
    target = ""
    for word in context.args:
        target += word
        target += " "

    if target in OLD_MP and random.random() < 0.7:
        update.message.reply_text(OLD_MP[target], quote=False)
    else:
        choice = random.choice(MP)
        OLD_MP[target] = choice
        update.message.reply_text(choice, quote=False)


def add_mp(update: Update, context: CallbackContext) -> None:
    new_mp = ""
    for word in context.args:
        new_mp += word
        new_mp += " "
    MP.append(new_mp)
    update.message.reply_text(new_mp, quote=False)


def delete_mp(update: Update, context: CallbackContext) -> None:
    delete_mp = ""
    for word in context.args:
        delete_mp += word
        delete_mp += " "
    if delete_mp in MP: MP.remove(delete_mp)
    update.message.reply_text(delete_mp, quote=False)


def get_article(update: Update, context: CallbackContext) -> None:
    link = "https://en.wikipedia.org/wiki/Special:Random"
    update.message.reply_text(link)


def puhu(update: Update, context: CallbackContext) -> None:
    print("call puhu")
    msg_str = update.message.text
    if msg_str == '':
        return
    
    conversation.add_message(msg_str)
    conv_messages = conversation.get_messages()

    return_str, _ = translator(conv_messages)

    if len(return_str) > 1:
        conversation.add_message(return_str)
        update.message.reply_text(return_str, quote=False)


def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('Commands: \n get, pajaa, mp, add_mp, delete_mp, artikkeli')


class Translator(tf.Module):
    def __init__(self, tokenizer, transformer, input_size=128, output_size=32):
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.input_size = input_size
        self.output_size = output_size
    
    def __call__(self, inputs, max_length=128):
        encoder_input = []
        for sentence in inputs:
            encoder_input += self.tokenizer(sentence)['input_ids'][1:-1]
      
        if len(encoder_input) > self.input_size:
            encoder_input = encoder_input[len(encoder_input)-self.input_size:]
        else:
            encoder_input = encoder_input + [tokenizer.pad_token_id] * self.input_size
            encoder_input = encoder_input[:self.input_size]

        encoder_input = tf.constant(encoder_input, dtype=tf.int64)[tf.newaxis]

        start_end = self.tokenizer([''])['input_ids'][0]
        start = tf.constant(start_end[0], dtype=tf.int64)[tf.newaxis]
        end = tf.constant(start_end[1], dtype=tf.int64)[tf.newaxis]
                
        output_array = tf.TensorArray(dtype=tf.int64, size=max_length, dynamic_size=True)
        output_array = output_array.write(0, start)
      
        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions, _ = self.transformer([encoder_input, output], training=False)
            predictions = predictions[:, i:i+1, :]
            
            pp = predictions.numpy().squeeze()
            for _ in range(500):
                pp = pp**5
                if np.count_nonzero(pp - pp.mean()) < 1:
                    break
                pp -= pp.mean()
                pp[pp < 0] = 0
                pp = np.power(pp, 1/5)

            print(np.count_nonzero(pp))
            if np.count_nonzero(pp) == 0:
                predicted_id = pp.argmax()
            else:
                pp = pp**5
                pp /= pp.sum()
                predicted_id = np.random.choice(np.arange(self.tokenizer.vocab_size), p=pp)
            output_array = output_array.write(i+1, np.expand_dims(predicted_id, 0))
            if predicted_id == end.numpy()[0]:
                break
            
        output = tf.squeeze(tf.transpose(output_array.stack()))
        text = self.tokenizer.decode(list(output), skip_special_tokens=True)
        tokens = self.tokenizer.convert_ids_to_tokens(output)
        return text, tokens

class Conversation():
    def __init__(self, length=5):
        self.length=length
        self.msgs = []
    
    def get_messages(self):
        return self.msgs

    def add_message(self, msg):
        self.msgs.append(msg)
        self.msgs = self.msgs[:self.length]

tokenizer = BertTokenizer.from_pretrained('../bert-marko-vocab.txt')
N_CLASSES = tokenizer.vocab_size

loaded_transformer = load_model('bert_kaikki_5')
translator = Translator(tokenizer, loaded_transformer)

conversation = Conversation(5)

def main() -> None:
    updater = Updater(TOKEN, use_context=True)
    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("get", get_time))
    dispatcher.add_handler(CommandHandler("pajaa", get_paja))
    dispatcher.add_handler(CommandHandler("mp", get_mp))
    dispatcher.add_handler(CommandHandler("add_mp", add_mp))
    dispatcher.add_handler(CommandHandler("delete_mp", delete_mp))
    dispatcher.add_handler(CommandHandler("artikkeli", get_article))
    dispatcher.add_handler(CommandHandler("help", help_command))

    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, puhu))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
