import nltk
import re
import heapq
import numpy as np

paragraph = """In her article, “I Ain’t Sayin’ He’s a Gold Digger”, author Hanna Rosin claims
that 40 percent of wives earn more than their husbands, and their husbands, as a
result, are unable to cope with the shift in this power dynamic. She provides the
testimony of “Andy,” a stay-at-home dad of twins, to exemplify this tension: “Andy
likes watching the toddlers, but he is wistful about his old life, and somewhat
defensive about his new one. These days when his wife suggests that he should go
back to work, Andy feels ‘terrified.’ It’s been a long time, and he’s lost the stomach
for the outside world.” In this example, Rosen illustrates the resentment that men
may feel toward their breadwinning wives. Rosin uses adjectives such as “wistful,”
“defensive,” and “terrified” to support her claim that men are insecure in their roles
at home, and have been robbed of their masculinity.
Andy Hinds, the stay-at-home dad whom Rosin interviewed, has responded
to her piece in his essay, “Hanna Rosin Turned Me into a Caricature.” Hinds claims
that Rosin greatly oversimplifies his situation. He clarifies the example she provides
by filling in more details: he works part-time teaching college English, taking on
small carpentry projects, and participating in an online community of blogging
fathers. He writes, “Of course I have fond memories and endless stories of manly
derring-do on the construction site, but these days I would far rather spend time
with my kids than with a bunch of smelly knuckleheads who think there’s nothing
funnier than to accuse one another of being gay all day long.” By providing more
detail of his life today, as contrasted to what transpired on a typical workday in his
“old life”, Hinds makes Rosin’s claim that he misses working construction sound
ridiculous. While Rosin might suggest that stay-at-home dads are lost without a
traditional job, Hinds complicates this claim. He reveals that there is more to his
life— and to his masculine identity—than being the primary breadwinner in his
home. If anything, Hinds suggests that he and his family are happy, and that his
situation is a way to make the “shifting gender paradigm” work."""
               
               
# Tokenize sentences
dataset = nltk.sent_tokenize(paragraph)
for i in range(len(dataset)):
    dataset[i] = dataset[i].lower()
    dataset[i] = re.sub(r'\W',' ',dataset[i])
    dataset[i] = re.sub(r'\s+',' ',dataset[i])

# Creating word histogram
word2count = {}
for data in dataset:
    words = nltk.word_tokenize(data)
    for word in words:
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1
            
# Selecting best 100 features
freq_words = heapq.nlargest(100,word2count,key=word2count.get)


# IDF Dictionary
word_idfs = {}
for word in freq_words:
    doc_count = 0
    for data in dataset:
        if word in nltk.word_tokenize(data):
            doc_count += 1
    word_idfs[word] = np.log(len(dataset)/(1+doc_count))
    
# TF Matrix
tf_matrix = {}
for word in freq_words:
    doc_tf = []
    for data in dataset:
        frequency = 0
        for w in nltk.word_tokenize(data):
            if word == w:
                frequency += 1
        tf_word = frequency/len(nltk.word_tokenize(data))
        doc_tf.append(tf_word)
    tf_matrix[word] = doc_tf
    
tfidf_matrix = []
for word in tf_matrix.keys():
    tfidf = []
    for value in tf_matrix[word]:
        score = value * word_idfs[word]
        tfidf.append(score)
    tfidf_matrix.append(tfidf)    
    
# Finishing the Tf-Tdf model
X = np.asarray(tfidf_matrix)

X = np.transpose(X)    