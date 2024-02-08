
import streamlit as st
import pandas as pd
import pinecone
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.memory import DynamoDBChatMessageHistory
import boto3

PINECONE_API_KEY = st.secrets.PINECONE_API_KEY
BEDROCK_REGION = st.secrets.AWS_DEFAULT_REGION
MAX_TOKENS = 10024
TEMPERATURE = 0.7

# Initialize clients and services
session = boto3.Session(region_name='us-east-1')
bedrock_client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
chat_history_DB = DynamoDBChatMessageHistory(table_name="SessionTable", session_id="1", boto3_session=session)
index_pinecone = 'unidosus-policy-test'
model_id = "anthropic.claude-v2:1"
model_kwargs = {"max_tokens_to_sample": MAX_TOKENS, "temperature": TEMPERATURE}
embeddings = BedrockEmbeddings(client=bedrock_client, region_name=BEDROCK_REGION)
llm = Bedrock(model_id=model_id, region_name=BEDROCK_REGION, client=bedrock_client, model_kwargs=model_kwargs)

def pinecone_db():
    """
    Initializes and returns the Pinecone index.
    """
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_pinecone)
    return index

def retrieval_answer(query, selected_years, types):
    """
    Retrieves answers and sources based on the query, selected years, and document types.
    """
    # Construct filter conditions for the query
    filter_conditions = create_filter_conditions(selected_years, types)
    index = pinecone_db()
    vectorstore = Pinecone(index, embeddings, "text")
    retriever = vectorstore.as_retriever(search_kwargs={'filter': filter_conditions, 'k': 20})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": f"{query}"})
    sources = render_search_results(response['context'])
    # Update chat history in DynamoDB
    chat_history_DB.add_user_message(query)
    ai_message = extract_answer_sources(response)
    chat_history_DB.add_ai_message(ai_message)
    return response['answer'], sources

def create_filter_conditions(selected_years, options=None):
    """
    Creates filter conditions for document retrieval based on selected years and types.
    """
    filter_conditions = {"year": {"$gte": selected_years[0], "$lte": selected_years[1]}}
    if options and "ALL" not in options:
        filter_conditions["type"] = {"$in": options}
    return filter_conditions


def render_search_results(documents):
    """
    Renders search results into a DataFrame for display.
    """
    metadata_list = []
    for doc in documents:
        # Obtenemos los metadatos básicos
        title = doc.metadata.get('title', '')
        source = doc.metadata.get('source', '').replace('s3://', 'https://s3.amazonaws.com/')
        doc_type = doc.metadata.get('type', '')
        year = doc.metadata.get('year', '')

        if year:
            year = str(int(year))

        metadata_list.append({"Title": title, "Source": source, "Type": doc_type, "Year": year})
    df = pd.DataFrame(metadata_list).drop_duplicates(subset=['Title'])
    return df

def extract_answer_sources(data):
    """
    Extracts and formats the answer and its sources from the response data.
    """
    answer = data.get('answer', '')
    sources = [document.metadata['source'] for document in data.get('context', []) if hasattr(document, 'metadata')]
    sources_str = "','".join(sources)
    result = f"{answer}source:'{sources_str}'."
    return result

# Define the prompt template for user queries
template = """Based on the context provided from the vector database, please perform the action as per the user's request:
<context>{context}</context>
Depending on the user's request, please:
- If the user asks for relevant documents on a specific topic, retrieve and display documents from the database that are pertinent to the topic.
- If the user requests a summary on a specific topic, provide a concise summary of the relevant information on the topic from the database.
User's Request: {input}"""
prompt = ChatPromptTemplate.from_template(template)
document_chain = create_stuff_documents_chain(llm, prompt)
