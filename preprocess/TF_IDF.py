import nltk
def tfidf():
    # from sklearn.feature_extraction.text import TfidfTransformer
    # from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    text = []
    i=0
    for line in open("cleanedNews.txt", "r"):
        i+=1
        text.append(line)
        # if i ==1000:
        #     break
    count = 0
    for corpus in text:
        # count+=1
        score = []
        context = nltk.word_tokenize(corpus)#词的list
        corpus1 = [str(corpus)]
        tfidf2 = TfidfVectorizer()
        re = tfidf2.fit_transform(corpus1)
        for name in context:
            if name in tfidf2.vocabulary_:
                j = tfidf2.vocabulary_[name]
                score.append(re.A[0][j])
            else:
                print("context:", context)
                print("tfidf2.vocabulary_", tfidf2.vocabulary_)
                print(count)
                print(name)
                break
        with open("tfidf1.txt", 'a') as file:
            for i in score:
                file.write(str(i)+" ")
            count+=1
            # print(count)
            file.write("\n")
tfidf()