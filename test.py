from dotenv import load_dotenv

load_dotenv()
from langchain.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage
from langchain.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

def get_tavily_answer_with_retry(query):
    # Set up the agent
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
    search = TavilySearchAPIWrapper()
    tavily_tool = TavilySearchResults(api_wrapper=search)

    # Initialize the agent
    agent_chain = initialize_agent(
        [tavily_tool],
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # Run the agent with user input question
    output = agent_chain.run(query)
    
    return output

if __name__ == "__main__":
    user_query = input("Enter your question: ")
    answer = get_tavily_answer_with_retry(user_query)
    print("Answer:", answer)

