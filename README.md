# context-aware-chatbot

link to streamlit deployed app: https://context-aware-chatbot-kz4yfyizgptbh9nrrcz4wm.streamlit.app/


This task contains a context-aware chatbot built using the LangChain framework, HuggingFace models, FAISS vector store, and Streamlit. It allows users to upload a '.txt' file as context and then ask questions based on the uploaded content.


Problem Statement:

The task is to build a 'context-aware chatbot' that can respond to user queries based on a custom text corpus provided at runtime. The chatbot must be able to understand context using vector embeddings and return meaningful, grounded responses without training a new model.


Task Objectives:

-  Allow users to upload a '.txt' file
-  Embed the text using HuggingFace Embeddings
-  Store and retrieve context using 'FAISS' vector database
-  Use a pre-trained model from HuggingFace ('google/flan-t5-small') for generating answers
-  Set up a 'RetrievalQA' pipeline using LangChain
-  Deploy the chatbot via 'Streamlit' with a clean and simple UI
-  Include a sample '.txt' file and fully working chatbot link


Tech Stack

| Tool            | Purpose                             |
|-----------------|-------------------------------------|
| Python          | Core programming language           |
| LangChain       | Framework for chaining LLM workflows|
| HuggingFace     | For embeddings & language model     |
| FAISS           | Vector similarity search            |
| Streamlit       | Interactive web app frontend        |


Features

- Upload any '.txt' document as a knowledge base
- Asks any question related to uploaded content
- Uses semantic understanding through embeddings
- Instant web-based UI with Streamlit

