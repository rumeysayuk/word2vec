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

# print(model.wv.most_similar("hollanda"))
# modele kelimeleri verip eğittik.Ülkeleri bağdaştırıp bize getirdi.

# print(model.wv.most_similar("pazartesi"))
# gün ile ilgili bağdaştırdıklarını getiriyor.

model.save("word2vec.model")
model = Word2Vec.load("word2vec.model")


def closeswords_tsneplot(model, word):
    word_vectors = np.empty((0, 100))
    word_labels = [word]

    close_words = model.wv.most_similar(word)
    word_vectors = np.append(word_vectors, np.array([model.wv[word]]), axis=0)
    # axis=0 verilmezse vektörler düzleştirilir.

    for w, _ in close_words:
        word_labels.append(w)
        word_vectors = np.append(word_vectors, np.array([model.wv[w]]), axis=0)
    tsne = TSNE(random_state=0)
    y = tsne.fit_transform(word_vectors)

    x_coords = y[:, 0]
    y_coords = y[:, 1]

    plt.scatter(x_coords, y_coords)
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(5, -2), textcoords="offset points")
    plt.show()


closeswords_tsneplot(model, "salı")
