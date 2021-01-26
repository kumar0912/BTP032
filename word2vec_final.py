global vocab_size
from itertools import chain
import nltk
import regex
import gensim
from gensim import corpora
from gensim.models.word2vec import Word2Vec
with open('/home/kumar/Desktop/Hindi_model/monolingual/monolingual.hi') as f:  
    text = f.read(10000000)
    text = regex.sub("(?s)<ref>.+?</ref>", "", text) # remove reference links
    text = regex.sub("(?s)<[^>]+>", "", text) # remove html tags
    text = regex.sub("&[a-z]+;", "", text) # remove html entities
    text = regex.sub("(?s){{.+?}}", "", text) # remove markup tags
    text = regex.sub("(?s){.+?}", "", text) # remove markup tags
    text = regex.sub("(?s)\[\[([^]]+\|)", "", text) # remove link target strings
    text = regex.sub("(?s)\[\[([^]]+\:.+?]])", "", text) # remove media links
    text = regex.sub("[']{5}", "", text) # remove italic+bold symbols
    text = regex.sub("[']{3}", "", text) # remove bold symbols
    text = regex.sub("[']{2}", "", text) # remove italic symbols
    text = regex.sub(u"[^ \r\n\p{Devanagari}.।?!\-]", " ", text)
    text = regex.sub("[ ]{2,}", " ", text) # Squeeze spaces.  
    head = regex.split(u"([.।?!])?[\n]+|[.।?!] ", text)
    
sen = []
for line in head:
    if line is None:
        continue
    else:    
        words = line.split()
        sen.append(words)
allwords = []
for l in sen:
    allwords += l

print(len(allwords))
print(len(set(allwords)))
    
"""
my_words = []
for word in allwords:
    my_words.append(t.generate_stem_words(word))

sen1 = []
for line in head:
    words = line.split()
    stem_words = []
    for word in words:
        new_word = t.generate_stem_words(word)
        #print(new_word)
        stem_words.append(new_word)
        #print(stem_words)
    sen1.append(stem_words)
"""
fdist = nltk.FreqDist(chain.from_iterable(sen))
min_count_obt = fdist.most_common(len(set(allwords)))[-1][1]

model = Word2Vec(sen, size=300, window=5, min_count=min_count_obt, negative=20, iter = 200)


model.save('saved_model/my_model_imp2')

print(len(allwords))
print(len(set(allwords)))



#for x in allwords:
#    print(x)

#for x in my_words:
#    print(x)


import os
import codecs
#import ic
import logging
import pandas as pd 
import numpy as np
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText
import gensim.models.wrappers.fasttext


# Log output. Also useful to show program is doing things
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# models trained using gensim implementation of word2vec

# Model Link: https://drive.google.com/drive/folders/1Ig-NVfWMGBRJqskfYF2aCh6d7ht-YwG4?usp=sharing
print('Loading models...')
model_source = KeyedVectors.load_word2vec_format(r'/home/kumar/Desktop/English_model/model.bin', binary = True)
#model_target = gensim.models.Word2Vec.load(r'/home/kumar/Desktop/saved_model/my_model_imp1')
model_target = model



# list of word pairs to train translation matrix as csv
# eg:
#  source,target
#  今日は、hello
#  犬、dog
#  猫、cat
print ('Reading training pairs...')

word_pairs = codecs.open(r'/home/kumar/Downloads/dataset1.csv', 'r', 'utf-8')

pairs = pd.read_csv(word_pairs)


print ('Removing missing vocabulary...')

missing = 0

for n in range (len(pairs)):
	if pairs['source'][n] not in model_source.vocab or pairs['target'][n] not in model_target.wv.vocab:
		missing = missing + 1
		pairs = pairs.drop(n)

pairs = pairs.reset_index(drop = True)
print('Amount of missing vocab: ', missing)

# make list of pair words, excluding the missing vocabs 
# removed in previous step
pairs['vector_source'] = [model_source[pairs['source'][n]] for n in range (len(pairs))]
pairs['vector_target'] = [model_target[pairs['target'][n]] for n in range (len(pairs))]




# first 5000 from both languages, to train translation matrix
source_training_set = pairs['vector_source'][:5000]
target_training_set = pairs['vector_target'][:5000]

matrix_train_source = pd.DataFrame(source_training_set.tolist()).values
matrix_train_target = pd.DataFrame(target_training_set.tolist()).values

print('Generating translation matrix')

# Matrix W is given in  http://stackoverflow.com/questions/27980159/fit-a-linear-transformation-in-python
translation_matrix = np.linalg.pinv(matrix_train_source).dot(matrix_train_target).T
print('Generated translation matrix')

# Returns list of topn closest vectors to vectenter
def most_similar_vector(self, vectenter, topn=5):
    self.init_sims()
    dists = np.dot(self.wv.syn0norm, vectenter)
    if not topn:
        return dists
    best = np.argsort(dists)[::-1][:topn ]
        # ignore (don't return) words from the input
    result = [(self.wv.index2word[sim], float(dists[sim])) for sim in best]
    return result[:topn]

def top_translations(w,numb=5):
    val = most_similar_vector(model_target,translation_matrix.dot(model_source[w]),numb)
    #print 'traducwithscofres ', val
    return val


def top_translations_list(w, numb=5):
    val = [top_translations(w,numb)[k][0] for k in range(numb)]
    return val

temp = 1
#top_matches = [ pairs['target'][n] in top_translations_list(pairs['source'][n]) for n in range(5000,5003)] 

# print out source word and translation
def display_translations():
    for word_num in range(range_start, range_end):
        source_word =  pairs['source'][word_num]
        translations = top_translations_list(pairs['source'][word_num]) 
        print (source_word, translations)

# range to use to check accuracy
range_start = 5000
range_end = 6000

display_translations()

# now we can check for accuracy on words 5000-6000, 1-5000 used to traning
# translation matrix

# returns matrix of true or false, true if translation is accuracy, false if not
# accurate means the first translation (most similiar vector in target language)
# is identical
accuracy_at_five = [pairs['target'][n] in top_translations_list(pairs['source'][n]) for n in range(range_start, range_end)]
print('Accuracy @5 is ', sum(accuracy_at_five), '/', len(accuracy_at_five))

accuracy_at_one = [pairs['target'][n] in top_translations_list(pairs['source'][n],1) for n in range(range_start, range_end)]
print('Accuracy @1 is ', sum(accuracy_at_one), '/', len(accuracy_at_one))

