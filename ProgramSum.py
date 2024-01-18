from pythainlp.corpus import thai_stopwords #ลบคำฟุ่มเฟือยภาษาไทย หรือ stopword เป็นคำที่ไม่ค่อยมีความสำคัญ เช่น คำเชื่อม คำอุทานต่างๆ (เป็น, คือ, เเล้ว)
thai_stopwords = list(thai_stopwords()) #เป็นการเรียกใช้ฟังก์ชัน thai_stopwords โดยเเปลงฟังก์ชันให้เป็นลิสต์ของคำเพื่อความง่ายในการเรียกใช้งาน
from pythainlp.tokenize import sent_tokenize #นำเข้าฟังก์ชันที่ใช้ในการตัดประโยค
from pythainlp.tokenize import word_tokenize #นำเข้าฟังก์ชันที่ใช้ในการตัดคำ
from string import punctuation

from pythainlp.tokenize import sent_tokenize
import nltk

# ตั้งค่า PyThaiNLP ให้ใช้ตัวตัดประโยคภาษาไทย
nltk.download('punkt')

from heapq import nlargest

import numpy as np
from tqdm.auto import tqdm
import torch
from functools import partial

#transformers
from transformers import (
    CamembertTokenizer,
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)

#thai2transformers
import thai2transformers
from thai2transformers.preprocess import process_transformers
from thai2transformers.metrics import (
    classification_metrics,
    multilabel_classification_metrics,
)
from thai2transformers.tokenizers import (
    ThaiRobertaTokenizer,
    ThaiWordsNewmmTokenizer,
    ThaiWordsSyllableTokenizer,
    FakeSefrCutTokenizer,
    SEFR_SPLIT_TOKEN
)


def word_freq(news):
  word_th = word_tokenize(news)

  word_freq_th = {} #1. สร้าง Dictionary
  for word in word_th: #2. เช็คคำที่ได้จากการตัดคำว่าต่างกับเงื่อนไขด้านล่างนี้หรือไม่ ถ้าตรงให้ค่าเป็น 1 ถือว่าเป็นคำเเรกที่เจอ เเต่ถ้าไม่ตรงให้บวกค่าเพิ่มเป็น 1 ถือว่าเป็นคำที่เคยเจอเเล้ว โดยเงื่อนไขที่ว่ามีดังนี้
    if word not in thai_stopwords: #เงื่อนไขที่ 1: มีคำที่เป็นคำในลิสต์ของคำ stopwords หรือไม่ ถ้าไม่ใช่ไปเช็คต่อที่เงื่อนไขที่ 2 ถ้าใช่ ไปที่ else เพื่อบวกค่าเพิ่มอีก 1 ค่า
      if word not in punctuation:  #เงื่อนไขที่ 2: มีเครื่องหมายวรรคตอนหรือไม่ !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ ถ้าไม่ใช่ไปเช็คต่อที่เงื่อนไขที่ 3 ถ้าใช่ ไปที่ else เพื่อบวกค่าเพิ่มอีก 1 ค่า
        if word not in " ": #เงื่อนไขที่ 3: มี white space หรือใหม่ ถ้าไม่ใช่ไปเช็คต่อที่เงื่อนไขที่ 4 ถ้าใช่ ไปที่ else เพื่อบวกค่าเพิ่มอีก 1 ค่า
          if word not in word_freq_th.keys(): #เงื่อนไขที่ 4: word นี้อยู่ใน key ของ dictionary word_freq_th ที่เราใช้เก็บค่าความถี่หรือไม่ถ้าไม่อยู่ ถือว่าตรงกับเงื่อนไขทั้งหมดให้เพิ่มค่าเป็น 1 ถ้าอยู่ถือว่าไม่ตรงให้ไปที่ else เเล้วบวกค่าเพิ่ม 1 ค่า
            word_freq_th[word] = 1 #เพิ่มค่าเป็น 1 เนื่องจากตรงกับทั้ง 4 เงื่อนไขที่ผ่านมา
          else:
            word_freq_th[word] += 1 #บวกค่าเพิ่มอีก 1 ค่า
  sorted(word_freq_th.items(), key=lambda x: x[1],reverse=True)

  #ทำการ normalize ความถี่ที่ได้เพื่อปรับช่วงของข้อมูลให้อยู่ในช่วงใกล้เคียงกัน
  max_freq_th = max(word_freq_th.values())
  for word in word_freq_th.keys():
    word_freq_th[word] = word_freq_th[word]/max_freq_th

  sorted(word_freq_th.items(), key=lambda x: x[1],reverse=True)

  return word_freq_th


def thai_sent_tokenize(text):
    # ใช้ PyThaiNLP เพื่อตัดประโยคภาษาไทย
    sentences = sent_tokenize(text)
    return sentences

def sentence(news):
  nltk.sent_tokenize = thai_sent_tokenize
  text_without_hashtag = news.replace("#", "")

  sent_th = nltk.sent_tokenize(text_without_hashtag)

  return sent_th


def sent_scores(sent_news, word_freq_th):
  sent_scores_th = {} #สร้าง dictionary
  for sent in sent_news: #นำประโยคที่ตัดไว้ทุกประโยคมาคำนวณ
    for word in sent: #เช็ค[คำ]ที่มีในประโยค A
      if word in word_freq_th.keys(): #ถ้าคำในประโยค A มีใน dictionary ของ word_freq_th(เก็บความถี่ของคำที่ตัดได้)
        if sent not in sent_scores_th.keys(): #ถ้าประโยคไม่ได้อยู่ใน dictionary ของ sent_scores_th
          sent_scores_th[sent] = word_freq_th[word] #ให้ sentence scores เท่ากับ ค่าความถี่ที่ normalize เเล้วของ word frequencies
        else:   #ถ้าประโยคอยู่ใน dictionary ของ sent_scores_th
          sent_scores_th[sent] += word_freq_th[word] #ให้ บวกเพิ่ม sentence scores เท่ากับ ค่าความถี่ที่ normalize เเล้วของ word frequencies ไปอีก 1 ครั้ง

  sorted(sent_scores_th.items(), key=lambda x: x[1], reverse=True)

  return sent_scores_th

'''
def result_freq(sent_scores_th):
  #select_len_th = int(len(sent_scores_th)*0.5) #*0.1 เป็นเลขที่ใช้ในการกำหนดความยาวของบทความที่สรุป หากลองเปลี่ยนเลขให้มากกว่า 0.1 (max=1)ความยาวเเละเนื้อหาที่สรุปก็จะมากขึ้น ขึ้นอยู่ที่ว่าเราต้องการให้บทความมีความสั้นยาวสรุปได้กระชับมากเเค่ไหน
  select_len_th = 7
  sum_th = nlargest(select_len_th, sent_scores_th, key=sent_scores_th.get) #เเสดงประโยคที่มีความสำคัญมากที่สุดจากค่า sentence scores โดยข้อมูลตัวที่ 1เเละ 2 จาก dict เนื่องจากความยาว len เท่ากับ 2

  #รวมข้อมูลตัวที่ 1 เเละ 2 ที่ได้จาก dict ให้อยู่ใน paragraph เดียวกัน
  sum_th = "".join(sum_th)

  return sum_th
'''
model_names = [
    'wangchanberta-base-att-spm-uncased',
    'xlm-roberta-base',
    'bert-base-multilingual-cased',
    'wangchanberta-base-wiki-newmm',
    'wangchanberta-base-wiki-ssg',
    'wangchanberta-base-wiki-sefr',
    'wangchanberta-base-wiki-spm',
]

tokenizers = {
    'wangchanberta-base-att-spm-uncased': AutoTokenizer,
    'xlm-roberta-base': AutoTokenizer,
    'bert-base-multilingual-cased': AutoTokenizer,
    'wangchanberta-base-wiki-newmm': ThaiWordsNewmmTokenizer,
    'wangchanberta-base-wiki-ssg': ThaiWordsSyllableTokenizer,
    'wangchanberta-base-wiki-sefr': FakeSefrCutTokenizer,
    'wangchanberta-base-wiki-spm': ThaiRobertaTokenizer,
}
public_models = ['xlm-roberta-base', 'bert-base-multilingual-cased']
#@title Choose Pretrained Model
model_name = "wangchanberta-base-att-spm-uncased" #@param ["wangchanberta-base-att-spm-uncased", "xlm-roberta-base", "bert-base-multilingual-cased", "wangchanberta-base-wiki-newmm", "wangchanberta-base-wiki-syllable", "wangchanberta-base-wiki-sefr", "wangchanberta-base-wiki-spm"]

#create tokenizer
tokenizer = tokenizers[model_name].from_pretrained(
                f'airesearch/{model_name}' if model_name not in public_models else f'{model_name}',
                revision='main',
                model_max_length=416,)

#pipeline
fill_mask = pipeline(task='fill-mask',
         tokenizer=tokenizer,
         model = f'airesearch/{model_name}' if model_name not in public_models else f'{model_name}',
         revision = 'main',)

#if the sequence is too short, it needs padding
def fill_mask_pad(input_text):
  return fill_mask(input_text+'<pad>')

def mask_sent(sent_scores_th):
  #select_len_th = int(len(sent_scores_th)*0.5)
  select_len_th = 7
  sum_th = nlargest(select_len_th, sent_scores_th, key=sent_scores_th.get)
  lenS=len(sum_th)-1
  re=[]
  for i in range(0,lenS):
    input_text = sum_th[i]+'<mask>'+sum_th[i+1]
    #print(input_text)
    preprocess_input_text = True #@param {type:"boolean"}
    if preprocess_input_text:
      if model_name not in public_models:
          input_text = process_transformers(input_text)

  #infer
    sen =fill_mask_pad(input_text)
    #print(sen)
    re.append(sum_th[i])
    re.append(sen[0]['token_str'])
    #print(re)

  re.append(sum_th[lenS])
  sum_th3 = "".join(re)
  newtext=sum_th3.replace("<_>", " ")

  return newtext


def title_sim(title,news):
  word_title = word_tokenize(title)

  sent_th = sentence(news)

  sent_th01 = sent_th
  sent_th01.append(title)

  sent_scores_title = {} #สร้าง dictionary
  for i in sent_th01:
    numS=0
    word_news = word_tokenize(i)
    for word in word_news:
      if word in word_title:
        numS+=1
    sim=numS/len(word_news)
    sent_scores_title[i] = sim

  sorted(sent_scores_title.items(), key=lambda x: x[1], reverse=True)

  return sent_scores_title

'''
def result_titleS(sent_scores_title):

  result_title = mask_sent(sent_scores_title)

  return result_title
'''

def sent_fr_sim(news, sent_scores_th, sent_scores_title):
  sent_scores_All = {}
  sent_th01 = sentence(news)

  for i in sent_th01:
    try:
      numAll = int(sent_scores_th[i]) - int(sent_scores_title[i])
      sent_scores_All[i] = numAll
    except:
      pass

  sorted(sent_scores_All.items(), key=lambda x: x[1], reverse=True)

  return sent_scores_All

def result_fr_sim(sent_scores_All):
  result_fr_sim = mask_sent(sent_scores_All)

  return result_fr_sim


def result(title,news_th):
    sent_scores_title4 = title_sim(title,news_th)

    #ตัดคำนับความถี่
    word_freq_th4 = word_freq(news_th)
    #ตัดประโยค
    sent_th4 = sentence(news_th)
    #คำนวณคะแนนจากความถี่
    sent_scores_th4 = sent_scores(sent_th4, word_freq_th4)

    #แสดงสรุป
    sent_scores_All4 = sent_fr_sim(news_th, sent_scores_th4, sent_scores_title4)
    sum_th4 = result_fr_sim(sent_scores_All4)
    
    return sum_th4
