from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from crewai_tools import BaseTool
from dotenv import load_dotenv
from googlesearch import search
from pydantic import Field

load_dotenv()


os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model='gemini-pro')

class WebSearchTool(BaseTool):
    name: str = Field(default="Web Search")
    description: str = Field(default="Search the web for information")

    def _run(self, query: str, num_results: int = 5) -> list[str]:
        results = []
        for url in search(query, num_results=num_results):
            results.append(url)
        return results
    
web_search_tool = WebSearchTool()

market_research_analyst = Agent(
        role="Market Research Analyst",
        goal="""Analyze the market demand for {product_name} and 
        suggest marketing strategies""",
        backstory="""Expert at understanding market demand, target audience, 
        and competition for products like {product_name}. 
        Skilled in developing marketing strategies 
        to reach a wide audience.""",
        verbose=True,
        allow_delegation=True,
        llm=llm,
        tools = [web_search_tool]
        )

technology_expert = Agent(
        role="Technology Expert",
        goal="Assess technological feasibilities and requirements for producing high-quality {product_name}",
        backstory="""Visionary in current and emerging technological trends, 
            especially in products like {product_name}. 
            Identifies which technologies are best suited 
            for different business models.""",
        verbose=True,
        allow_delegation=True,
        llm=llm,
        tools = [web_search_tool]
        )

business_consultant = Agent(
        role="Business Development Consultant",
        goal="""Evaluate the business model for {product_name}, 
        focusing on scalability and revenue streams""",
        backstory="""Seasoned in shaping business strategies for products like {product_name}. 
            Understands scalability and potential 
            revenue streams to ensure long-term sustainability.""",
        verbose=True,
        allow_delegation=True,
        llm=llm,
        tools = [web_search_tool]
        )


# Define Tasks
task1 = Task(
description="""Analyze the market demand for {product_name}. Current month is June 2024.
    Write a report on the ideal customer profile and marketing 
    strategies to reach the widest possible audience. 
    Include at least 10 bullet points addressing key marketing areas.""",
expected_output="Report on market demand analysis and marketing strategies.",
agent=market_research_analyst,
)
# Define Task 2
task2 = Task(
description="""Assess the technological aspects of manufacturing 
high-quality {product_name}. Write a report detailing necessary 
technologies and manufacturing approaches. 
Include at least 10 bullet points on key technological areas.""",
expected_output="Report on technological aspects of manufacturing.",
agent=technology_expert,
)
# Define Task 3
task3 = Task(
description="""Summarize the market and technological reports 
and evaluate the business model for {product_name}. 
Write a report on the scalability and revenue streams 
for the product. Include at least 10 bullet points 
on key business areas. Give Business Plan, 
Goals and Timeline for the product launch. Current month is Jan 2024.""",
expected_output="Report on business model evaluation and product launch plan.",
agent=business_consultant,
output_file = 'output.md'
)

# Create and Run the Crew
product_crew = Crew(
agents=[market_research_analyst, technology_expert, business_consultant],
tasks=[task1, task2, task3],
verbose=2,
process=Process.sequential,
)

def start_crew(product_name):
    crew_result = product_crew.kickoff(inputs={'product_name': product_name})
    return crew_result
