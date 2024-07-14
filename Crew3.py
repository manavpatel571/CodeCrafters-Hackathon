import os
from dotenv import load_dotenv
from crewai import Task, Agent, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Set the Google API key from the environment variable
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model='gemini-pro')

# Define the Planning Agent
PlanningAgents = Agent(
    role='Senior System Analyst',
    goal='Expand the project description by getting more information from the user.',
    backstory="""
        You are an expert in system analysis.
        You are able to communicate effectively with both technical and non-technical stakeholders.
        This includes active listening, asking questions, and explaining technical concepts in simple terms.
        You are able to process and interpret the information from stakeholders to identify patterns, trends, and gaps in the information.
        You are able to assess the validity and reliability of the sources.
        You are attentive to details and able to identify inconsistencies in the information provided.
        You are always thorough in your analysis to find any and all the information required to properly define a software development project.
        You enjoy generating the most accurate and complete project descriptions possible.
        You will cover all the most important aspects of the project, including but not limited to:
            - the project's goals and objectives;
            - major features;
            - constraints, assumptions, and risks;
            - target audience;
            - security requirements, line authentication, authorization, and data protection;
            - data requirements, like data sources, data formats, data storage, data access;
            - design preferences, like colors, fonts, themes, layouts, navigation;
            - development requirements, like programming languages, frameworks, and tools;
        You will analyze the current information about the project and ask the user for more details to refine the project description.
        You will keep asking until you have all the information you need to properly define the project or until the user asks you to proceed.
    """,
    memory=True,
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Define the tasks
research_task = Task(
    description="""
        Conduct comprehensive research on each of the individuals and companies
        involved in the upcoming meeting. Gather information on recent
        news, achievements, professional background, and any relevant
        business activities.

        Participants: {participants}
        Meeting Context: {context}""",
    expected_output="""
        A detailed report summarizing key findings about each participant
        and company, highlighting information that could be relevant for the meeting.""",
    agent=PlanningAgents
)

industry_analysis_task = Task(
    description="""
        Analyze the current industry trends, challenges, and opportunities
        relevant to the meeting's context. Consider market reports, recent
        developments, and expert opinions to provide a comprehensive
        overview of the industry landscape.

        Participants: {participants}
        Meeting Context: {context}""",
    expected_output="""
        An insightful analysis that identifies major trends, potential
        challenges, and strategic opportunities.""",
    agent=PlanningAgents
)

meeting_strategy_task = Task(
    description="""
        Develop strategic talking points, questions, and discussion angles
        for the meeting based on the research and industry analysis conducted.
        Meeting Context: {context}
        Meeting Objective: {objective}""",
    expected_output="""
        Complete report with a list of key talking points, strategic questions
        to ask to help achieve the meetings objective during the meeting.""",
    agent=PlanningAgents
)

summary_and_briefing_task = Task(
    description="""
        Compile all the research findings, industry analysis, and strategic
        talking points into a concise, comprehensive briefing document for
        the meeting.
        Ensure the briefing is easy to digest and equips the meeting
        participants with all necessary information and strategies.

        Meeting Context: {context}
        Meeting Objective: {objective}""",
    expected_output="""
        A well-structured briefing document that includes sections for
        participant bios, industry overview, talking points, and
        strategic recommendations.""",
    agent=PlanningAgents,
    output_file = 'meeting.md'
)

# Define the crew
crew = Crew(
    agents=[PlanningAgents],
    tasks=[research_task, industry_analysis_task, meeting_strategy_task, summary_and_briefing_task],
    process=Process.sequential,
    verbose=True
)

def start_crew3(participants, context, objective):
    crew_result = crew.kickoff(inputs={'participants': participants, 'context': context, 'objective': objective})
    return crew_result
