 
import re
import numpy as np
import random
import nltk
import string
from contextlib2 import suppress
from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
from spellchecker import SpellChecker
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')
nltk.download('maxent_ne_chunker')
f= open(r"D:\downloads\WhatsApp Chat with newme (4).txt", 'r',errors = 'ignore')
raw_doc=f.read()
raw_doc = raw_doc.lower()
sentence_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)
print(sentence_tokens)
def recognize_names_with_nltk(text):
  with suppress(UserWarning):
    sentences = sent_tokenize(text)
    words = [word_tokenize(sent) for sent in sentences]

    tagged_words = [pos_tag(word) for word in words]

    named_entities = []
    for tagged_sentence in tagged_words:
        chunked_sentence = ne_chunk(tagged_sentence, binary=False)
        for subtree in chunked_sentence:
            if type(subtree) == nltk.Tree:
                entity = " ".join([word for word, tag in subtree.leaves()])
                label = subtree.label()
                named_entities.append((entity, label))

    return named_entities

recognize_names_with_nltk(raw_doc)
lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
  return [lemmer.lemmatize(token) for token in tokens]
remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
  return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))
import re

greet_inputs = re.compile(r'(?i)(hello|hi|whatsup|how are you|hey|good evening|greetings|howdy|salutations)')

def greet(sentence):
  match = greet_inputs.search(sentence)
  if match:
    greeting = random.choice(["Hello", "Hi", "Hey", "Hey there","Greetings","Aloha","Good Afternoon"])
    return greeting.format(user_name="User")
  else:
    return None
with suppress(UserWarning):
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.metrics.pairwise import cosine_similarity
def response(user_response):
  with suppress(UserWarning):
    bot_response = ""
    TfidfVec = TfidfVectorizer(tokenizer = LemNormalize, stop_words = 'english')
    tfidf = TfidfVec.fit_transform(sentence_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if (req_tfidf == 0):
        bot_response = bot_response + "I am sorry. Unable to understand you!"
        return bot_response
    else:
        bot_response = bot_response + sentence_tokens[idx]
        return bot_response
spell = SpellChecker()

def correct_typos(text):
    words = text.split()
    corrected_words = []

    for word in words:
        corrected_word = spell.correction(word)
        corrected_words.append(corrected_word)

    corrected_text = ' '.join(corrected_words)
    return corrected_text

flag= True
print("Hello! I am the learning Bot. Start typing your text after greeting to talk to me. For ending convo type bye")
while(flag == True):
  with suppress(UserWarning):
    user_response = input()
    user_response = user_response.lower()
    user_response = correct_typos(user_response)
    if(user_response !='bye'):
      if(user_response =='thank you' or user_response == 'thanks'):
        flag = True
        print("Bot: You are welcome, can I help you with any thing else?")
      elif(user_response =='yes'):
        flag = True
        print("Elaborate your problem please")
      elif(user_response =='no' or user_response == 'no,thanks'):
        flag = False
        print("Alright")
        
      else:
        if(greet(user_response) != None):
          print('Bot: ' +greet(user_response))
        else:
          sentence_tokens.append(user_response)
          word_tokens = word_tokens + nltk.word_tokenize(user_response)
          final_words = list(set(word_tokens))
          print('Bot: ', end = '')
          print(response(user_response))
          sentence_tokens.remove(user_response)
          
          