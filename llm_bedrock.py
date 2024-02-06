import streamlit as st
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import pinecone 
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import BedrockEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import AmazonKendraRetriever
from langchain.llms.bedrock import Bedrock
import boto3
import toml


PINECONE_API_KEY = st.secrets.PINECONE_API_KEY
PINECONE_ENV = st.secrets.PINECONE_ENV
bedrock_region = st.secrets.AWS_BEDROCK_REGION
max_tokens = 1024  # Adjust as needed
temperature = 0.7  # Adjust as needed
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
index_pinecone  = 'unidosus-policy-test'
model_id = "anthropic.claude-v2:1"
model_kwargs = {"max_tokens_to_sample": max_tokens, "temperature": temperature}
embeddings = BedrockEmbeddings(client=bedrock_client, region_name="us-east-1")
# Setup bedrock
llm = Bedrock(model_id=model_id, region_name=bedrock_region, client=bedrock_client, model_kwargs=model_kwargs)


def pinecone_db():
    # we use the openAI embedding model
    index_name = index_pinecone
    #strat the Pinecone Index
    pc = pinecone.Pinecone(
    api_key=PINECONE_API_KEY)
    index = pc.Index(index_name)
    return index

def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

template_old = """"Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}
"""


template = """"Based on the context provided from the vector database, please perform the action as per the user's request:

<context>
{context}
</context>

Depending on the user's request, please:
- If the user asks for relevant documents on a specific topic, retrieve and display documents from the database that are pertinent to the topic.
- If the user requests a summary on a specific topic, provide a concise summary of the relevant information on the topic from the database.
User's Request: {input}
"""

prompt = ChatPromptTemplate.from_template(template)
document_chain = create_stuff_documents_chain(llm, prompt)

# Function to retrieve answers
def retrieval_answer(query, selected_years ):        
    #filter_condition = {"page": {"$eq": 1}}
    filter_condition_year = {
    "year": {
        "$gte": selected_years[0],  # Mayor o igual al año de inicio
        "$lte": selected_years[1]   # Menor o igual al año de finalización
    }
    }
    # Select the model based on user choice
    index = pinecone_db()
    vectorstore = Pinecone(index, embeddings, "text")
    retriever = vectorstore.as_retriever(search_kwargs={'filter': filter_condition_year, 'k': 20})
    #retriever = vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": f"{query}"})
    sources = render_search_results(response['context'])
    chat_history_DB.add_user_message(query)
    chat_history_DB.add_ai_message(response['answer'])
    return response['answer'], sources 


def render_search_results(documents):
    # Initialize a list to hold metadata dictionaries
    metadata_list = []

    # Iterate over each Document object in the documents list
    for doc in documents:
        # Assuming each doc item is an object with a metadata attribute
        # that behaves like a dictionary, directly access its properties
        metadata = doc.metadata  # Adjust this line if accessing metadata is different
        metadata_entry = {
            'Title': metadata.get('title', ''),
            'Source': metadata.get('source', ''),
            'Type': metadata.get('type', ''),
            'Year': metadata.get('year', '')
        }
        # Append the extracted metadata to our list
        metadata_list.append(metadata_entry)
    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(metadata_list)
    df = df.drop_duplicates(subset=['Title'])
    return df


from langchain.memory import DynamoDBChatMessageHistory
session = boto3.Session(region_name='us-east-1')

bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
chat_history_DB = DynamoDBChatMessageHistory(table_name="SessionTable", session_id="991", boto3_session=session)

def update_memory(human_input, ai_output):
    """
    Update the global memory with new conversation turns.
    """
    #chat_history_DB.save_context({'input': human_input}, {'output': ai_output})
    chat_history_DB.add_user_message(human_input)
    chat_history_DB.add_ai_message(ai_output)

    return "Memory updated"



