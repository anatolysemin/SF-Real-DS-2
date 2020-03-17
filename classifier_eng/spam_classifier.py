import pandas as pd
import math
import string
 
pA, pNotA = 0, 0
SPAM, NOT_SPAM = 1, 0
spam_list, not_spam_list = {}, {}
total_spam, total_not_spam = 0,0

def get_words(body):
    first = body.lower()
    second = str.maketrans(dict.fromkeys(string.punctuation))
    third = first.translate(second)
    body = third.split()
    return body

def calculate_word_frequencies(body, label):
    global spam_list, not_spam_list, total_spam, total_not_spam
    words = get_words(body)
    for word in words:
        if label ==  SPAM:
            spam_list[word] = spam_list.get(word, 0) + 1
            total_spam += 1
        else:
            not_spam_list[word] = not_spam_list.get(word, 0) + 1
            total_not_spam += 1

def train():
    global pA, pNotA
    df = pd.read_csv('spam_or_not_spam.csv')
    df.email.iloc[-34] = 'number'
    train_data, lbl = [], []
    for email in df.email:
        train_data.append([email])
    for label in df.label:
        lbl.append(label)
    for index,text in enumerate(train_data):
        text.append(lbl[index])
    num_spam = 0
    for words,label in train_data:
        calculate_word_frequencies(words, label)
        if label == SPAM:
            num_spam += 1
    pA =  num_spam / len(train_data)
    pNotA = 1 - pA

def calculate_P_Bi_A(word, label):
    if label == SPAM:
        return (spam_list.get(word, 0) + 1) / total_spam
    return (not_spam_list.get(word, 0) + 1) / total_not_spam

def calculate_P_B_A(body, label):
    P_B_A = 1
    for word in get_words(body):
        log_P_Bi_A = math.log(calculate_P_Bi_A(word, label))
        P_B_A  += log_P_Bi_A #  здесь используем "+", а не "*", т.к. применяем логариф http://bazhenov.me/blog/2012/06/11/naive-bayes.html
    return P_B_A

def classify(email):   
    P_A_B = math.log(pA) + calculate_P_B_A(email, SPAM) #  здесь используем "+", а не "*", т.к. применяем логариф
    P_A_not_B = math.log(pNotA) + calculate_P_B_A(email, NOT_SPAM)
    return P_A_B > P_A_not_B

