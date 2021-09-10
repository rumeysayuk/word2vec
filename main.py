import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

f = open("hurriyet.txt", "r", encoding="utf8")  # r =Read   utf8 turkçe karakterlerle çalışcağımızı belirttik.
text = f.read()
t_list = text.split('\n')

corpus = []
for sentence in t_list:
    a = corpus.append(sentence.split())
print(corpus[:10])

model = Word2Vec(corpus, vector_size=100, window=5, min_count=5, sg=1)
print(model)
# sg=1 skip-gram alg kullanılacak demek.default cbow kullanılıyor

# print(model.wv["ankara"])

print(model.wv.most_similar("hollanda"))
# modele kelimeleri verip eğittik.Ülkeleri bağdaştırıp bize getirdi.

print(model.wv.most_similar("pazartesi"))
# gün ile ilgili bağdaştırdıklarını getiriyor.

model.save("word2vec.model")
model = Word2Vec.load("word2vec.model")
