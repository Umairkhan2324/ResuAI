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
    
    # Create a single-task crew
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential
    )
    
    return crew.kickoff()

def get_next_question(agent, field_analysis: Dict, previous_responses: Dict) -> str:
    task = Task(
        description="""
        Based on the job field analysis and previous responses, generate the next most relevant interview question.
        Consider what information is still needed for a complete resume.
        """,
        expected_output="A clear, focused question for the candidate",
        agent=agent,
        context=[
            f"Job Field Analysis: {field_analysis}",
            f"Previous Responses: {previous_responses}"
        ]
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential
    )
    
    return crew.kickoff()

def build_resume(agent, data: Dict) -> str:
    task = Task(
        description="Create a professional resume using the provided data",
        expected_output="A well-formatted HTML resume",
        agent=agent,
        context=[
            f"Resume Data: {data}"
        ]
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential
    )
    
    return crew.kickoff()

def get_feedback(agent, resume: str) -> str:
    task = Task(
        description="Review the resume and provide detailed, actionable feedback",
        expected_output="Comprehensive feedback with specific improvement suggestions",
        agent=agent,
        context=[
            f"Resume Content: {resume}"
        ]
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential
    )
    
    return crew.kickoff()

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

class DynamicInterviewer:
    def __init__(self, llm):
        self.agent = Agent(
            role='Professional Interviewer',
            goal='Gather detailed professional information through adaptive questioning',
            backstory='Experienced recruiter skilled in extracting relevant career information',
            llm=llm,
            verbose=True
        )
        
        self.required_topics = [
            'personal_info',
            'professional_summary',
            'work_experience',
            'education',
            'skills',
            'projects'
        ]
        
        self.questions = {
            'personal_info': "Please provide your personal information (name, email, phone number).",
            'professional_summary': "Could you provide a brief summary of your professional background and career goals?",
            'work_experience': "Please detail your work experience, including company names, positions, and key achievements.",
            'education': "What is your educational background? Include degrees, institutions, and graduation years.",
            'skills': "What are your key technical and professional skills?",
            'projects': "Could you describe any significant projects you've worked on?"
        }
        
        self.current_topic_index = 0

    def get_next_question(self, field_analysis: Dict, previous_responses: Dict) -> str:
        if not previous_responses:
            return self.questions['personal_info']
            
        # Determine which topic to ask about next
        for topic in self.required_topics[self.current_topic_index:]:
            # Check if we have enough information about this topic
            if not self._has_topic_info(topic, previous_responses):
                return self.questions[topic]
            self.current_topic_index += 1
            
        # If we've covered all basic topics, ask follow-up questions
        return self._generate_followup_question(field_analysis, previous_responses)

    def _has_topic_info(self, topic: str, responses: Dict) -> bool:
        # Check if any response contains keywords related to the topic
        topic_keywords = {
            'personal_info': ['name', 'email', 'phone'],
            'professional_summary': ['summary', 'background', 'objective'],
            'work_experience': ['work', 'job', 'position', 'company'],
            'education': ['education', 'degree', 'university', 'school'],
            'skills': ['skills', 'abilities', 'competencies'],
            'projects': ['project', 'portfolio', 'achievement']
        }
        
        keywords = topic_keywords.get(topic, [])
        responses_text = ' '.join(responses.values()).lower()
        return any(keyword in responses_text for keyword in keywords)

    def _generate_followup_question(self, field_analysis: Dict, previous_responses: Dict) -> str:
        task = Task(
            description="Generate a follow-up question based on previous responses",
            expected_output="A specific follow-up question to gather more detailed information",
            agent=self.agent,
            context=[
                f"Field Analysis: {field_analysis}",
                f"Previous Responses: {previous_responses}"
            ]
        )
        
        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            process=Process.sequential
        )
        
        return crew.kickoff()

    def is_interview_complete(self, responses: Dict) -> bool:
        # Check if we have information for all required topics
        return all(self._has_topic_info(topic, responses) for topic in self.required_topics)

def create_resume(agent, responses: Dict) -> Dict:
    resume_task = Task(
        description="Create a professional HTML resume",
        expected_output="HTML formatted resume",
        agent=agent,
        context=[f"Interview Responses: {responses}"]
    )
    
    feedback_task = Task(
        description="Provide detailed feedback on the resume",
        expected_output="Actionable feedback for improvement",
        agent=agent,
        context=[f"Interview Responses: {responses}"]
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[resume_task, feedback_task],
        process=Process.sequential
    )
    
    result = crew.kickoff()
    
    # Parse results
    try:
        if "<html" in result and "</html>" in result:
            html_start = result.find("<html")
            html_end = result.find("</html>") + 7
            resume_html = result[html_start:html_end]
            feedback = result[html_end:].strip()
        else:
            resume_html = f"<html><body>{result}</body></html>"
            feedback = "No specific feedback available."
        
        return {
            "resume": resume_html,
            "feedback": feedback if feedback else "No feedback provided."
        }
    except Exception as e:
        return {
            "resume": f"<html><body><p>Error: {str(e)}</p></body></html>",
            "feedback": "Error generating resume and feedback."
        }

if __name__ == "__main__":
    result = run_crew("Artificial Intelligence")
    print(result) 