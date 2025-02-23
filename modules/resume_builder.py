from jinja2 import Environment, FileSystemLoader
from typing import Dict
from crewai import Agent, Task

class ResumeBuilder:
    def __init__(self, llm):
        self.agent = Agent(
            role='Resume Expert',
            goal='Create outstanding, ATS-friendly resumes',
            backstory="""I am an expert resume writer who creates compelling,
            ATS-optimized resumes that highlight candidates' strengths and achievements.
            I know how to present information effectively for different industries.""",
            llm=llm,
            verbose=True
        )
        self.env = Environment(loader=FileSystemLoader('templates'))

    def create_resume(self, job_field: str, info: Dict) -> str:
        task = Task(
            description=f"""
            Create a professional resume for {job_field}...
            """,
            expected_output="HTML formatted resume with feedback",
            agent=self.agent
        )
        return task.execute_sync()

    def build_resume(self, data: Dict, template: str = 'resume_template.html') -> str:
        template = self.env.get_template(template)
        return template.render(**data) 