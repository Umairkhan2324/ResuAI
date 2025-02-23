from crewai import Agent
from typing import Dict
from crewai.tasks import Task

class DynamicInterviewer:
    def __init__(self, llm):
        self.agent = Agent(
            role='Professional Interviewer',
            goal='Gather comprehensive professional information through natural conversation',
            backstory="""I am an expert interviewer who knows how to ask the right questions
            to gather detailed information for resumes. I adapt my questions based on the job field
            and previous responses to ensure all relevant information is collected.""",
            llm=llm,
            verbose=True
        )

    def gather_information(self, job_field: str, current_info: Dict) -> Dict:
        task = Task(
            description=f"""
            Given the job field: {job_field}...
            """,
            expected_output="Follow-up questions based on missing information",
            agent=self.agent
        )
        return task.execute() 