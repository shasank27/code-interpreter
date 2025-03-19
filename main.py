import os 
import dotenv
dotenv.load_dotenv()
from langchain.tools import tool, Tool
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent

def main():
    print("Start")
    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    tools = [PythonREPLTool()]
    agent = create_react_agent(
        prompt=prompt,
        llm=ChatGoogleGenerativeAI(temperature= 0, model="gemini-2.0-flash"),
        tools=tools,
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # agent_executor.invoke(
    #     input={
    #         "input": """generate and save in qr_code directory 10 QRcodes
    #                             that point to www.udemy.com/course/langchain, you have qrcode package installed already"""
    #     }
    # )

    csv_agent = create_csv_agent(
        llm = ChatGoogleGenerativeAI(temperature= 0, model="gemini-2.0-flash"),
        path = "episode_info.csv",
        verbose = True,
        allow_dangerous_code=True
    )
    # csv_agent.invoke(input={"input": "How many columns are there in file episode_info.csv"})
    # csv_agent.invoke(input={"input": "In the file episode_info, which writer wrote the most episodes? How many episodes did he write? Split the writers if there are two or more writers in the same episode."})

    # ROUTER GRAND AGENT

    tools = [
        Tool(
            name="Python Agent",
            func=agent_executor.invoke,
            description="""useful when you need to transform natural language to python and execute the python code,
                            returning the results of code execution
                            DOES NOT ACCEPT CODE AS INPUT"""
        ),
        Tool(
            name="CSV Agent",
            func=csv_agent.invoke,
            description="""useful when you need to answer question over episode_info.csv file,
                            takes an input the entire question and returns the answer after running pandas calculation"""
        )
    ]

    prompt = base_prompt.partial(instructions="")
    grand_agent = create_react_agent(
        prompt=prompt,
        llm=ChatGoogleGenerativeAI(temperature= 0, model="gemini-2.0-flash"),
        tools=tools
    )
    grand_agent_executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True)
    # print(grand_agent_executor.invoke(input={"input": "Which season has the most episode?"}))
    print(grand_agent_executor.invoke(input={"input": "Generate 12 QR code that point to instagram.com/shasankperiwal, you have qrcode package installed already"}))

if __name__ == "__main__":
    main()