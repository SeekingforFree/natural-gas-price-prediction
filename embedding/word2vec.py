import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)

# 返回一个词 的向量：
print(model['word'])

# 返回和一个词语最相关的多个词语以及对应的相关度
items = model.most_similar('happy')
for item in items:
    # 词的内容，词的相关度
    print(item[0], item[1])

print(y)