from crewai import Agent, Task, Crew, Process, LLM
import google.generativeai as genai
from config import GEMINI_API_KEY
from typing import Dict, List, Optional
from pydantic import BaseModel
from jinja2 import Environment, FileSystemLoader

class ResumeData(BaseModel):
    personal_info: dict
    professional_summary: str
    work_experience: List[dict]
    education: List[dict]
    skills: List[str]
    projects: Optional[List[dict]]

def initialize_llm():
    # Configure Gemini
    genai.configure(api_key=GEMINI_API_KEY)
    return LLM(
        model="gemini/gemini-1.5-pro-latest",
        temperature=0.5,
    )

def create_job_analyzer(llm):
    return Agent(
        role='Job Field Analysis Expert',
        goal='Analyze job fields and identify key requirements',
        backstory='Expert in job market analysis with deep understanding of industry requirements',
        llm=llm,
        verbose=True
    )

def create_interviewer(llm):
    return Agent(
        role='Professional Interviewer',
        goal='Gather detailed professional information through adaptive questioning',
        backstory='Experienced recruiter skilled in extracting relevant career information',
        llm=llm,
        verbose=True
    )

def create_resume_builder(llm):
    return Agent(
        role='Resume Creation Specialist',
        goal='Create professional and ATS-optimized resumes',
        backstory='Expert resume writer with extensive experience in professional document creation',
        llm=llm,
        verbose=True
    )

def create_feedback_agent(llm):
    return Agent(
        role='Resume Review Specialist',
        goal='Provide actionable feedback on resumes',
        backstory='Professional resume reviewer with expertise in multiple industries',
        llm=llm,
        verbose=True
    )

def analyze_job_field(agent, job_field: str) -> Dict:
    task = Task(
        description=f"Analyze the {job_field} field and identify key requirements, skills, and keywords.",
        expected_output="A dictionary containing industry keywords, ATS requirements, and template suggestions",
        agent=agent
    )
    return task.execute()

def conduct_interview(agent, job_field: str, field_analysis: Dict) -> Dict:
    task = Task(
        description=f"Conduct an adaptive interview for a {job_field} position based on initial information",
        expected_output="Detailed professional profile including skills, experience, and achievements",
        agent=agent
    )
    return task.execute()

def build_resume(agent, data: ResumeData) -> str:
    task = Task(
        description="Create a professional resume using the provided data",
        expected_output="A well-formatted HTML resume",
        agent=agent
    )
    return task.execute()

def get_feedback(agent, resume: str) -> str:
    task = Task(
        description="Review the resume and provide detailed, actionable feedback",
        expected_output="Comprehensive feedback with specific improvement suggestions",
        agent=agent
    )
    return task.execute()

def run_crew(job_field: str):
    llm = initialize_llm()
    
    # Create agents
    job_analyzer = create_job_analyzer(llm)
    interviewer = create_interviewer(llm)
    resume_builder = create_resume_builder(llm)
    feedback_agent = create_feedback_agent(llm)
    
    # Create tasks
    analyze_task = Task(
        description=f"Analyze the {job_field} field requirements",
        expected_output="Job field analysis results in JSON format",
        agent=job_analyzer
    )
    
    interview_task = Task(
        description="Conduct professional interview and format the response as HTML",
        expected_output="HTML formatted resume content with proper sections",
        agent=interviewer
    )
    
    resume_task = Task(
        description="""
        Create a professional resume in HTML format with the following structure:
        - Professional Summary
        - Work Experience
        - Education
        - Skills
        - Projects (if applicable)
        Include appropriate HTML tags and styling.
        """,
        expected_output="Complete HTML resume with proper formatting and structure",
        agent=resume_builder
    )
    
    feedback_task = Task(
        description="Review the resume and provide actionable feedback",
        expected_output="Detailed feedback with specific improvement suggestions",
        agent=feedback_agent
    )
    
    # Create and run crew
    crew = Crew(
        agents=[job_analyzer, interviewer, resume_builder, feedback_agent],
        tasks=[analyze_task, interview_task, resume_task, feedback_task],
        process=Process.sequential
    )
    
    result = crew.kickoff()
    
    # Split the result into resume and feedback sections
    try:
        # Assuming the LLM will format the output with clear separators
        if "<html" in result and "</html>" in result:
            html_start = result.find("<html")
            html_end = result.find("</html>") + 7
            resume_html = result[html_start:html_end]
            feedback = result[html_end:].strip()
        else:
            # Fallback if HTML tags aren't found
            resume_html = f"<html><body>{result}</body></html>"
            feedback = "No specific feedback available."
            
        return {
            "resume": resume_html,
            "feedback": feedback if feedback else "No feedback provided."
        }
    except Exception as e:
        return {
            "resume": f"<html><body><p>Error processing resume: {str(e)}</p></body></html>",
            "feedback": "Error occurred while generating feedback."
        }

if __name__ == "__main__":
    result = run_crew("Artificial Intelligence")
    print(result) 