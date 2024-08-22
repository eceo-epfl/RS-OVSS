import random
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
import pandas as pd
import string

from nltk.corpus import wordnet, stopwords

#[EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks]
# (https://aclanthology.org/D19-1670) 
# (Wei & Zou, EMNLP-IJCNLP 2019)

# ========================== Synonym Replacement ========================== #
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    if word in synonyms:
        synonyms.remove(word)
    
    return list(synonyms)

def synonym_replacement(sentence, n):
    
    words = sentence.split()

    # Replace "..." with your code
    
    # Remove stopword from the list of potential word to replace :
    stop_words = stopwords.words('english')
    random_word_list = [word.lower() for word in words if word.lower() not in stop_words]    

    # Verify that the potential words to replace have a synonyms :
    random_word_list = [word for word in random_word_list if len(wordnet.synsets(word))>=1]  
    
    
    # Sample random  words to replace by their synonyms 
    # sample k words that is the minimum between n words or number of words with synonyms :
    random_word_list = random.sample( random_word_list, k = min(n,len(random_word_list)))

    new_sentence = sentence
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms)<1 : # no synonyms found 
            continue
        new_sentence = new_sentence.replace(random_word, random.choice (synonyms ))
    

    return new_sentence


# ========================== Random Deletion ========================== #
def random_deletion(sentence, p, max_deletion_n):

    words = sentence.split()
    max_deletion_n = min(max_deletion_n, len(words)-1)
    
    # obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words # sentence ?

    # Replace "..." with your code
    new_sentence = words
    deletion_n = 0
    
    for idx, word in enumerate (words) :
        if random.uniform(0,1)<p :
            new_sentence.remove(word)
            deletion_n+=1
            
        if deletion_n>= max_deletion_n:
            break
    new_sentence = ' '.join(new_sentence)           
    
    return new_sentence


# ========================== Random Swap ========================== #
def swap_word(sentence):
    words = sentence.split()
    if len(words) <= 1:
      return sentence

    # Replace "..." with your code
    random_idx_1, random_idx_2 = random.sample(range(len(words)),k=2)
    new_sentence = words
    new_sentence[random_idx_1], new_sentence[random_idx_2] = words[random_idx_2],words[random_idx_1]
    new_sentence = ' '.join(new_sentence)


    return new_sentence

# ========================== Random Insertion ========================== #
def random_insertion(sentence, n):
    
    words = sentence.split()
    new_words = words.copy()
    
    for _ in range(n):
        add_word(new_words)
        
    new_sentence = ' '.join(new_words)
    return new_sentence

def add_word(new_words):
    
    synonyms = []

    max_it=10
    while len(synonyms) <1 :
        synonyms = get_synonyms(random.choice( new_words )) 
        max_it -=1
        if max_it ==0:
            return
    random_synonym = random.choice(synonyms)
    new_words.insert( random.randint(0,len(new_words)-1), random_synonym)
    

def aug(sent,n,p):
    print(f" Original Sentence : {sent}")
    print(f" SR Augmented Sentence : {synonym_replacement(sent, n)}")
    print(f" RD Augmented Sentence : {random_deletion(sent, p, n)}")
    print(f" RS Augmented Sentence : {swap_word(sent)}")
    print(f" RI Augmented Sentence : {random_insertion(sent,n)}")
    



def augment_sentences(sent):
    # Apply randomly one of the four methods to augment the sentences :
    p = random.random()    

    nb = len(sent)//10
    aug_sent = sent
    if nb  > 0 :
        aug_sent = synonym_replacement(sent, nb)
    if  random.random()  <0.1:
        aug_sent = swap_word(sent )    
    if  random.random()  <0.1:
        aug_sent = random_insertion(sent ,1)  
    if  random.random()  <0.1:
        aug_sent = random_deletion(sent , 0.1, 1) # deletion limited to one word, can be break the sentence semantics
    return aug_sent


def clean_sentence(sentence):
    cleaned_sentence = sentence.lower()
    PUNK = string.punctuation
    cleaned_sentence = ''.join([x for x in cleaned_sentence if x not in string.punctuation])


    return cleaned_sentence
    
