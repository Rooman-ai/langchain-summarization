import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from utils.summarizer import summarizer_chain

load_dotenv()



deployment = os.getenv("DEPLOYMENT_NAME")



llm = AzureChatOpenAI(
    azure_deployment=deployment,
   
    temperature=0  
)





loader = TextLoader("text_files/ai_intro.txt",encoding="utf-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20,separator="")
docs = text_splitter.split_documents(documents)


embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("deployment"),  
    
   
)
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


query = "AI milestones"
retrieved_docs = retriever.get_relevant_documents(query)
retrieved_text = " ".join([doc.page_content for doc in retrieved_docs])


summarizer_chain(retrieved_text,llm)
