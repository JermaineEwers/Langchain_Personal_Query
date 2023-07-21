import os
import sys

from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
#from vectorstore import VectorStore

#from vectorstore import TextLoader, VectorstoreIndexCreator, VectorStore
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import LLMMathChain
from langchain.agents import Tool
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.indexes import VectorstoreIndexCreator
import vectorstore


os.environ["OPENAI_API_KEY"]= "PUT YOUR PERSONAL API KEY HERE"
prompt = PromptTemplate(
    input_variables=["query"],
    template="{query}"
)

llm = OpenAI(
    openai_api_key="PUT YOUR PERSONAL API KEY HERE",
    temperature=0,
    model_name="text-davinci-003"
    
)


#TOOLS


query = sys.argv[1]
loader = TextLoader('data.txt')
index = VectorstoreIndexCreator().from_loaders([loader])
llm_chain = LLMChain(llm=llm, prompt=prompt)

llm_math = LLMMathChain(llm=llm)
math_tool= Tool(
    name='Calculator',
    func=llm_math.run,
    description='Useful for when you need to answer questions about math.'
)

llm_tool = Tool(
    name='Language Model',
    func=llm_chain.run,
    #func=llm_chain.run,
    description='use this tool for general purpose queries and logic'
 
)





tools=load_tools(
    ['llm-math'],
    llm=llm
)

tools.append(llm_tool)




zero_shot_agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3
)

        
    print(zero_shot_agent(sys.argv[1]))

