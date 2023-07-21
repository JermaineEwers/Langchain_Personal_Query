import os
import sys

from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory





os.environ["OPENAI_API_KEY"]= "YOUR OWN API KEY"
query = sys.argv[1]
loader = TextLoader('data.txt')
#loader =  DirectoryLoader(".",glob="data.txt") 
index = VectorstoreIndexCreator().from_loaders([loader])





print(index.query(query,llm=OpenAI))
