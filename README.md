# LLM Book Recommendation Chatbot

This repository hosts a Language Model (LLM) based book recommendation chatbot built using Streamlit, Hugging Face's Datasets library, OpenAI's text-embedding-3-small model for text embedding, and MongoDB as the database. The chatbot allows users to ask for book recommendations and provides personalized responses based on their input.

## Features

- **Book Recommendation:** Users can ask for book recommendations by providing input to the chatbot.
- **Personalized Responses:** The chatbot provides personalized recommendations based on the user's input.
- **Streamlit Dashboard:** The chatbot is deployed using Streamlit, providing a user-friendly interface for interaction.
- **Data Source:** Book recommendation data is sourced from the [Book Recommendation Dataset](https://huggingface.co/datasets/egecandrsn/book_recommendation) available on Hugging Face's Datasets library.
- **Text Embedding:** OpenAI's text-embedding-3-small model is utilized for text embedding, enabling the chatbot to understand and respond to user queries effectively.
- **Database Integration:** MongoDB is used as the database for storing and retrieving book recommendation data.

## Usage

To run the chatbot locally, follow these steps:

1. Clone the repository:
git clone https://github.com/shreyas-kalikar/LLM-Book-Recommendation-Chatbot.git

2. Set up MongoDB and ensure it's running locally or provide connection details.

3. Run the Streamlit app:
   !npx localtunnel --port 8501 & streamlit run app.py & curl ipv4.icanhazip.com
   
5. Access the chatbot interface in your web browser at `http://localhost:8501`.

## Acknowledgements

- [Hugging Face](https://huggingface.co/) for providing the Book Recommendation Dataset and the Datasets library.
- [OpenAI](https://openai.com/) for the text-embedding-3-small model.
- [MongoDB](https://www.mongodb.com/) for providing the database technology.
