from crewai import Agent, Task
from typing import Dict

class DynamicInterviewer:
    def __init__(self, llm):
        self.agent = Agent(
            role='Professional Interviewer',
            goal='Gather detailed professional information through adaptive questioning',
            backstory='Experienced recruiter skilled in extracting relevant career information',
            llm=llm,
            verbose=True
        )

    def conduct_interview(self, job_field: str, initial_info: Dict) -> Dict:
        task = Task(
            description=f"Conduct an adaptive interview for a {job_field} position based on initial information",
            expected_output="Detailed professional profile including skills, experience, and achievements",
            agent=self.agent
        )

        return task.execute() 