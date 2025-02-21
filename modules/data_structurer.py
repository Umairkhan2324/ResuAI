from pydantic import BaseModel
from typing import List, Optional
from crewai import Agent, Task

class ResumeData(BaseModel):
    personal_info: dict
    professional_summary: str
    work_experience: List[dict]
    education: List[dict]
    skills: List[str]
    projects: Optional[List[dict]]

class DataStructurer:
    def __init__(self, llm):
        self.agent = Agent(
            role='Data Structure Specialist',
            goal='Convert raw resume data into structured format',
            backstory='Expert in data organization and standardization',
            llm=llm,
            verbose=True
        )

    def structure_data(self, raw_data: dict) -> ResumeData:
        task = Task(
            description="Structure and validate the raw resume data",
            expected_output="Standardized resume data in JSON format",
            agent=self.agent
        )
        
        structured_data = task.execute()
        return ResumeData(**structured_data) 