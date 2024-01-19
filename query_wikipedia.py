import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import WikipediaLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import os


def load_from_wikipedia(query, lang='en', load_max_docs=2):
    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
    data = loader.load()
    return data

def chunk_data(data, chunk_size=256):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks

def insert_or_fetch_embeddings(index_name):
    

    embeddings = OpenAIEmbeddings()

    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))

    if index_name in pinecone.list_indexes():
        print(f'Index {index_name} already exists. Loading embeddings ... ', end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('Ok')
    else:
        print(f'Creating index {index_name} and embeddings ...', end='')
        pinecone.create_index(index_name, dimension=1536, metric='cosine')
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        print('Ok')

    return vector_store

def delete_pinecone_index(index_name='all'):
    
    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))

    if index_name == 'all':
        indexes = pinecone.list_indexes()
        print('Deleting all indexes ... ')
        for index in indexes:
            pinecone.delete_index(index)
        print('Ok')
    else:
        print(f'Deleting index {index_name} ...', end='')
        pinecone.delete_index(index_name)
        print('Ok')

def ask_with_memory(vector_store, question, chat_history=[]):
    

    llm = ChatOpenAI(temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})

    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = crc({'question': question, 'chat_history': chat_history})
    chat_history.append((question, result['answer']))

    return result, chat_history

def __main__():
    data = load_from_wikipedia('', 'de')
    print(data[0].page_content)
    chunks = chunk_data(data)
    print(len(chunks))
    index_name = 'askadocument'
    vector_store = insert_or_fetch_embeddings(index_name)
    q = ''
    answer = ask_with_memory(vector_store, q)
    print(answer)        