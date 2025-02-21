from jinja2 import Environment, FileSystemLoader
from typing import Dict
from crewai import Agent

class ResumeBuilder:
    def __init__(self, llm):
        self.agent = Agent(
            role='Resume Creation Specialist',
            goal='Create professional and ATS-optimized resumes',
            backstory='Expert resume writer with extensive experience in professional document creation',
            llm=llm,
            verbose=True
        )
        self.env = Environment(loader=FileSystemLoader('templates'))

    def build_resume(self, data: Dict, template: str = 'resume_template.html') -> str:
        template = self.env.get_template(template)
        return template.render(**data) 