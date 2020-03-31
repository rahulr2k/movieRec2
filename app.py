from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import string
#import re
import pandas as pd
import numpy as np
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
def token(text):
    tokenized_word=word_tokenize(text)
    return tokenized_word
movies = pd.read_csv('MPR.csv')
print("tokenization started")
movies.keywords = movies.keywords.astype(str).apply(token)
print("tokenization completed , starting similarity engine")
processed_keywords = movies.keywords

movies.originalTitle  = movies.originalTitle.astype(str).apply(lambda x : x.replace("'", ''))
titlelist = movies.originalTitle.values.tolist()


from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel

with open('tfidf3.pkl', 'rb') as f:
        tfidf = pickle.load(f) #create tfidf model of the corpus

with open('dictionary3.pkl', 'rb') as f:
    dictionary = pickle.load(f)

with open('tfidfcorpus3.pkl', 'rb') as f:
        tfidfcorpus = pickle.load(f) 

 
        #create corpus where the corpus is a bag of words for each document
corpus = [dictionary.doc2bow(doc) for doc in processed_keywords] 

from gensim.similarities import MatrixSimilarity
        # Create the similarity data structure. This is the most important part where we get the similarities between the movies.
sims = MatrixSimilarity(tfidfcorpus, num_features=len(dictionary))  

print("similarity generation completed")

app = Flask(__name__)
 
@app.route('/')
def home():
        
    return render_template('home.html',prediction = titlelist)



@app.route('/predict',methods=['POST'])
def predict():
    
    
    if request.method == 'POST':

        message = request.form.get('message')
                
      
        ###### helper functions. Use them when needed #######
        
        def get_poster_from_index(index):
            return movies[movies.originalTitle == index]["poster"].values[0]
        def get_url_from_index(index):
            return movies[movies.originalTitle == index]["URL"].values[0]

        


        ##################################################


        #count_matrix = pickle.load(open("vector (3).pkl", 'rb'))
        ##Step 5: Compute the Cosine Similarity based on the count_matrix
        
        

        from gensim.corpora.dictionary import Dictionary
        from gensim.models.tfidfmodel import TfidfModel

        with open('tfidf3.pkl', 'rb') as f:
             tfidf = pickle.load(f) #create tfidf model of the corpus

        with open('dictionary3.pkl', 'rb') as f:
            dictionary = pickle.load(f)

        with open('tfidfcorpus3.pkl', 'rb') as f:
             tfidfcorpus = pickle.load(f) 

 
        #create corpus where the corpus is a bag of words for each document
        corpus = [dictionary.doc2bow(doc) for doc in processed_keywords] 

        from gensim.similarities import MatrixSimilarity
        # Create the similarity data structure. This is the most important part where we get the similarities between the movies.
        sims = MatrixSimilarity(tfidfcorpus, num_features=len(dictionary))   
        

        movie_title = message
        number_of_hits=5
        movie = movies.loc[movies.originalTitle==movie_title] # get the movie row
        keywords = movie.keywords.iloc[0] #get the keywords as a Series (movie['keywords']),
        # get just the keywords string ([0]), and then convert to a list of keywords (.split(',') )
        query_doc = keywords #set the query_doc to the list of keywords
        query_doc_bow = dictionary.doc2bow(query_doc) # get a bag of words from the query_doc
        query_doc_tfidf = tfidf[query_doc_bow] #convert the regular bag of words model to a tf-idf model where we have tuples
            # of the movie ID and it's tf-idf value for the movie

        similarity_array = sims[query_doc_tfidf] # get the array of similarity values between our movie and every other movie. 
            #So the length is the number of movies we have. To do this, we pass our list of tf-idf tuples to sims.
        similarity_series = pd.Series(similarity_array.tolist(), index=movies.originalTitle.values) #Convert to a Series
        top_hits = similarity_series.sort_values(ascending=False)[1:number_of_hits+1] 
            #get the top matching results, i.e. most similar movies; start from index 1 because every movie is most similar to itself

            #print the words with the highest tf-idf values for the provided movie:
        sorted_tfidf_weights = sorted(tfidf[corpus[movie.index.values.tolist()[0]]], key=lambda w: w[1], reverse=True)
        
 
        movie0 = []
        for idx, (movie,score) in enumerate(zip(top_hits.index, top_hits)):
            movie0.append(movie)

        ## Step 7: Get a list of similar movies in descending order of similarity score
        #sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)


        movie1 = []
        movie2 = []
        
        for element in movie0:
            
            movie1.append(get_url_from_index(element))
            movie2.append(get_poster_from_index(element))
            



        
        

        


    return render_template('result.html',movie0=movie0,movie1=movie1,movie2=movie2)


if __name__ == '__main__':
    app.run(debug=True)