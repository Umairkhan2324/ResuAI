from crewai import Agent, Task
from typing import Dict
from crewai_tools import WebsiteSearchTool

web_rag_tool = WebsiteSearchTool()
class JobFieldAnalyzer:
    def __init__(self, llm):
        self.agent = Agent(
            role='Job Field Analysis Expert',
            goal='Analyze job fields and identify key requirements',
            backstory='Expert in job market analysis with deep understanding of industry requirements',
            llm=llm,
            tools=[web_rag_tool],
            verbose=True
        )

    def analyze_field(self, job_field: str) -> Dict:
        task = Task(
            description=f"Analyze the {job_field} field and identify key requirements, skills, and keywords.",
            expected_output="A dictionary containing industry keywords, ATS requirements, and template suggestions",
            agent=self.agent
        )

        result = task.execute()
        # Process the result into a structured format
        return {
            'keywords': [],  # Extract keywords from result
            'ats_requirements': [],  # Extract ATS requirements
            'template_suggestion': 'default'  # Suggest appropriate template
        } 