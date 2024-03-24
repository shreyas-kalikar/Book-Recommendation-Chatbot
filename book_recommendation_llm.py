# -*- coding: utf-8 -*-


!pip install datasets pandas openai pymongo

# 1. Load Dataset
from datasets import load_dataset
import pandas as pd

# Taken and edited from https://huggingface.co/datasets/egecandrsn/book_recommendation
dataset = pd.read_csv("books.csv")

# Convert the dataset to a pandas dataframe
dataset_df = pd.DataFrame(dataset)

dataset_df.head(5)

print("Columns:", dataset_df.columns)
print("\nNumber of rows and columns:", dataset_df.shape)
print("\nBasic Statistics for numerical data:")
print(dataset_df.describe())
print("\nNumber of missing values in each column:")
print(dataset_df.isnull().sum())

#Dropping the embedding columns,i.e., image_embeddings and description_embeddings
dataset_df = dataset_df.drop(columns=['image_embeddings'])
dataset_df = dataset_df.drop(columns=['description_embeddings'])

dataset_df.columns

import openai
from google.colab import userdata

openai.api_key = "<YOUR-OPEN-AI-KEY>"

EMBEDDING_MODEL = "text-embedding-3-small"

def get_embedding(text):
    ###Generate an embedding for the given text using OpenAI's API.

    # Check for valid input
    if not text or not isinstance(text, str):
        return None

    try:
        # Call OpenAI API to get the embedding
        embedding = openai.embeddings.create(input=text, model=EMBEDDING_MODEL).data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error in get_embedding: {e}")
        return None

dataset_df["desc_embedding_optimised"] = dataset_df['description'].apply(get_embedding)

dataset_df.head()

import pymongo
from google.colab import userdata

def get_mongo_client(mongo_uri):
  ###Establish connection to the MongoDB.
  try:
    client = pymongo.MongoClient(mongo_uri)
    print("Connection to MongoDB successful")
    return client
  except pymongo.errors.ConnectionFailure as e:
    print(f"Connection failed: {e}")
    return None

mongo_uri = "<MONGO-DB-URL-HERE>"
if not mongo_uri:
  print("MONGO_URI not set in environment variables")

mongo_client = get_mongo_client(mongo_uri)

# Ingest data into MongoDB
db = mongo_client['books']
collection = db['books_collection']

# Delete any existing records in the collection
collection.delete_many({})

documents = dataset_df.to_dict('records')
collection.insert_many(documents)

print("Data ingestion into MongoDB completed")

def vector_search(user_query, collection):
    
    # Perform a vector search in the MongoDB collection based on the user query.

    # Args:
    # user_query (str): The user's query string.
    # collection (MongoCollection): The MongoDB collection to search.

    # Returns:
    # list: A list of matching documents.
    

    # Generate embedding for the user query
    query_embedding = get_embedding(user_query)

    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    # Define the vector search pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "plot_embedding_optimised",
                "numCandidates": 150,  # Number of candidate matches to consider
                "limit": 5  # Return top 5 matches
            }
        },
        {
            "$project": {
                "Book-Title": 1, # Include the Title field
                "score": {
                    "$meta": "vectorSearchScore"  # Include the search score
                }
            }
        }
    ]

    # Execute the search
    results = collection.aggregate(pipeline)
    return list(results)

def handle_user_query(query, collection):

  get_knowledge = vector_search(query, collection)

  search_result = ''
  for result in get_knowledge:
      search_result += f"Book Title: {result.get('Book-Title', 'N/A')}, Description: {result.get('description', 'N/A')}\n"

  completion = openai.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
          {"role": "system", "content": "You are a book recommendation system."},
          {"role": "user", "content": "Answer this user query: " + query + " with the following context: " + search_result}
      ]
  )

  return (completion.choices[0].message.content), search_result


!npm install localtunnel
!pip install streamlit

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# import openai
# from google.colab import userdata
# import pymongo
# import streamlit as st
# 
# openai.api_key = "<YOUR-OPEN-AI-KEY>"
# EMBEDDING_MODEL = "text-embedding-3-small"
# 
# 
# def get_mongo_client(mongo_uri):
#   """Establish connection to the MongoDB."""
#   try:
#     client = pymongo.MongoClient(mongo_uri)
#     print("Connection to MongoDB successful")
#     return client
#   except pymongo.errors.ConnectionFailure as e:
#     print(f"Connection failed: {e}")
#     return None
# 
# mongo_uri = "<MONGO-DB-URL-HERE>"
# if not mongo_uri:
#   print("MONGO_URI not set in environment variables")
# 
# mongo_client = get_mongo_client(mongo_uri)
# 
# # Ingest data into MongoDB
# db = mongo_client['books']
# collection = db['books_collection']
# 
# # Delete any existing records in the collection
# def get_embedding(text):
#     """Generate an embedding for the given text using OpenAI's API."""
# 
#     # Check for valid input
#     if not text or not isinstance(text, str):
#         return None
# 
#     try:
#         # Call OpenAI API to get the embedding
#         embedding = openai.embeddings.create(input=text, model=EMBEDDING_MODEL).data[0].embedding
#         return embedding
#     except Exception as e:
#         print(f"Error in get_embedding: {e}")
#         return None
# 
# def vector_search(user_query, collection):
#     """
#     Perform a vector search in the MongoDB collection based on the user query.
# 
#     Args:
#     user_query (str): The user's query string.
#     collection (MongoCollection): The MongoDB collection to search.
# 
#     Returns:
#     list: A list of matching documents.
#     """
# 
#     # Generate embedding for the user query
#     query_embedding = get_embedding(user_query)
# 
#     if query_embedding is None:
#         return "Invalid query or embedding generation failed."
# 
#     # Define the vector search pipeline
#     pipeline = [
#         {
#             "$vectorSearch": {
#                 "index": "vector_index",
#                 "queryVector": query_embedding,
#                 "path": "plot_embedding_optimised",
#                 "numCandidates": 150,  # Number of candidate matches to consider
#                 "limit": 5  # Return top 5 matches
#             }
#         },
#         {
#             "$project": {
#                 "Book-Title": 1, # Include the genres field
#                 "score": {
#                     "$meta": "vectorSearchScore"  # Include the search score
#                 }
#             }
#         }
#     ]
# 
#     # Execute the search
#     results = collection.aggregate(pipeline)
#     return list(results)
# 
# def handle_user_query(query, collection):
# 
#   get_knowledge = vector_search(query, collection)
# 
#   search_result = ''
#   for result in get_knowledge:
#       search_result += f"Book Title: {result.get('Book-Title', 'N/A')}, Description: {result.get('description', 'N/A')}\n"
# 
#   completion = openai.chat.completions.create(
#       model="gpt-3.5-turbo",
#       messages=[
#           {"role": "system", "content": "You are a book recommendation system."},
#           {"role": "user", "content": "Answer this user query: " + query + " with the following context: " + search_result}
#       ]
#   )
# 
#   return (completion.choices[0].message.content), search_result
# 
# 
# 
# st.set_page_config(page_title="Book Recommendation")
# st.header("Ask for a Book Recommendation!")
# query = st.text_input("Query")
# if query != "":
#   response, source_information = handle_user_query(query, collection)
#   print(source_information)
#   print(response)
#   st.write(f"Response: {response}\n\nSource Information: {source_information}")
#



