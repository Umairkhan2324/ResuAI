from jinja2 import Environment, FileSystemLoader
from typing import Dict
from crewai import Agent

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
        return self.agent.execute_task(
            f"""
            Create a professional resume for {job_field} using:
            {info}
            
            Format in clean HTML and include:
            1. Contact information
            2. Professional summary
            3. Work experience
            4. Education
            5. Skills
            6. Projects/Achievements
            
            Add feedback after ---FEEDBACK--- marker
            
            Format as:
            <RESUME>[HTML resume]---FEEDBACK---[Your feedback]</RESUME>
            """
        )

    def build_resume(self, data: Dict, template: str = 'resume_template.html') -> str:
        template = self.env.get_template(template)
        return template.render(**data) 