from crewai import Agent, Task
from typing import Dict

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
            Given the job field: {job_field}
            And current information: {current_info}
            
            Ask relevant questions to gather missing information about:
            1. Professional experience
            2. Education and certifications
            3. Technical and soft skills
            4. Projects and achievements
            
            Format response as:
            <QUESTIONS>[Your follow-up questions]</QUESTIONS>
            """,
            expected_output="Follow-up questions based on missing information",
            agent=self.agent
        )
        return task.execute_sync() 