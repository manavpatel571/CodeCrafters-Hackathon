from crewai import Agent, Task, Process, Crew
from crewai_tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from googlesearch import search
from pydantic import Field
import logging

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)

llm = ChatGoogleGenerativeAI(model='gemini-pro', verbose=True, temperature=0.5)

class WebSearchTool(BaseTool):
    name: str = Field(default="Web Search")
    description: str = Field(default="Search the web for information")

    def _run(self, query: str, num_results: int = 4) -> list[str]:
        results = []
        for url in search(query, num_results=num_results):
            results.append(url)
            logging.debug(f"WebSearchTool results for query '{query}': {results}")
        return results

web_search_tool = WebSearchTool()

# Create a senior content researcher
researcher = Agent(
    role='Senior Researcher',
    goal='Get relevant and accurate information for the user query {topic} from provided tools.',
    name='Senior Content Researcher',
    description='A senior content researcher who excels in finding and synthesizing information from Provided Tools.',
    backstory="An expert in finding and synthesizing information from various sources.",
    allow_delegation=True,
    tools=[web_search_tool],
    llm=llm
)

# Creating a senior writer agent
writer = Agent(
    role='Senior Writer',
    goal='Narrate compelling details about the topic {topic} from tools and llm model.',
    name='Senior Strategist Writer',
    description='A senior strategist writer skilled in creating actionable strategies based on data.',
    backstory="Skilled in creating actionable strategies based on data.",
    allow_delegation=False,
    tools=[web_search_tool],
    llm=llm
)

# Research Task
research_task = Task(
    description=(
        "Identify the user prompt {topic} use tools. "
        "Get detailed information about the user query."
    ),
    expected_output='A comprehensive and detailed accurate report based on the {topic}.',
    agent=researcher,
    tools = [web_search_tool]
)

# Write Task
write_task = Task(
    description=(
        "Get the information from the user query {topic}. "
        "Summarize the information and create content for accurate results."
    ),
    expected_output='Summarize the information on the user query and create content for the user.',
    agent=writer,
    tools = [web_search_tool],
    output_file='Output.txt'
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    max_rpm=10,
    embedder={
        "provider": "google",
        "config": {
            "model": 'models/embedding-001',
        }
    }
)

def start_crew(topic):
    result = crew.kickoff(inputs={'topic': topic})
    return result

