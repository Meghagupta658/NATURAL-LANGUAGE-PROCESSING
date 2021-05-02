import nltk

para = "Krishna Apra Garden is my sweet home"

words = nltk.word_tokenize(para)
    
tagged_word = nltk.pos_tag(words)
    
namedEnt =nltk.ne_chunk(tagged_word)
namedEnt.draw()