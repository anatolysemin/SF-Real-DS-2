import math
import string
 
pA, pNotA = 0, 0
SPAM, NOT_SPAM = 1, 0
spam_list, not_spam_list = {}, {}
total_spam, total_not_spam = 0,0
 
train_data = [  
    ['Купите новое чистящее средство', SPAM],   
    ['Купи мою новую книгу', SPAM],  
    ['Подари себе новый телефон', SPAM],
    ['Добро пожаловать и купите новый телевизор', SPAM],
    ['Привет давно не виделись', NOT_SPAM], 
    ['Довезем до аэропорта из пригорода всего за 399 рублей', SPAM], 
    ['Добро пожаловать в Мой Круг', NOT_SPAM],  
    ['Я все еще жду документы', NOT_SPAM],  
    ['Приглашаем на конференцию Data Science', NOT_SPAM],
    ['Потерял твой телефон напомни', NOT_SPAM],
    ['Порадуй своего питомца новым костюмом', SPAM]
]  

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
    train_data = [  
        ['Купите новое чистящее средство', SPAM],   
        ['Купи мою новую книгу', SPAM],  
        ['Подари себе новый телефон', SPAM],
        ['Добро пожаловать и купите новый телевизор', SPAM],
        ['Привет давно не виделись', NOT_SPAM], 
        ['Довезем до аэропорта из пригорода всего за 399 рублей', SPAM], 
        ['Добро пожаловать в Мой Круг', NOT_SPAM],  
        ['Я все еще жду документы', NOT_SPAM],  
        ['Приглашаем на конференцию Data Science', NOT_SPAM],
        ['Потерял твой телефон напомни', NOT_SPAM],
        ['Порадуй своего питомца новым костюмом', SPAM]
    ]  
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

