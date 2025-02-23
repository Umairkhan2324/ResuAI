from crewai import Agent, Task, Crew, Process, LLM
from crewai.project import CrewBase, agent, crew, task
from models import (
    PersonalInfo, ProfessionalInfo, JobAnalysis, ResumeData,
    ConversationMessage, AgentResponse
)
from typing import Dict, List
from datetime import datetime

def initialize_llm():
    return LLM(
        model="gemini/gemini-1.5-pro-latest",
        temperature=0.1
    )

@CrewBase
class ResumeCrewAI:
    """Resume creation crew with specialized agents"""
    
    def __init__(self):
        self.conversation_history: List[ConversationMessage] = []
        self.resume_data = ResumeData(
            job_field="",
            personal_info=PersonalInfo(),
            professional_info=ProfessionalInfo(),
            job_analysis=JobAnalysis(
                field="",
                required_skills=[],
                soft_skills=[],
                qualifications=[],
                industry_trends=[],
                career_paths=[],
                ats_keywords=[]
            )
        )
        self.llm = initialize_llm()

    def add_to_history(self, role: str, message: str) -> None:
        self.conversation_history.append(
            ConversationMessage(role=role, message=message)
        )

    def get_context(self) -> str:
        return "\n".join([
            f"{msg.role}: {msg.message}"
            for msg in self.conversation_history[-5:]
        ])

    @agent
    def supervisor(self) -> Agent:
        return Agent(
            role='Supervisor',
            goal='Coordinate resume creation process',
            backstory="""Expert career advisor coordinating resume creation through
            analysis and delegation to specialized agents.""",
            llm=self.llm,
            verbose=True
        )

    @agent
    def interviewer(self) -> Agent:
        return Agent(
            role='Professional Interviewer',
            goal='Gather comprehensive professional information',
            backstory="""Expert interviewer gathering detailed career information
            through adaptive questioning.""",
            llm=self.llm,
            verbose=True
        )

    @agent
    def job_analyzer(self) -> Agent:
        return Agent(
            role='Job Market Analyst',
            goal='Analyze job requirements and market trends',
            backstory="""Industry expert providing insights on requirements,
            trends, and career paths.""",
            llm=self.llm,
            verbose=True
        )

    @agent
    def resume_builder(self) -> Agent:
        return Agent(
            role='Resume Expert',
            goal='Create ATS-optimized resumes',
            backstory="""Expert resume writer creating compelling,
            ATS-friendly resumes.""",
            llm=self.llm,
            verbose=True
        )

    @task
    def analyze_situation(self) -> Task:
        return Task(
            description=f"""
            Current Context: {[msg.dict() for msg in self.conversation_history[-5:]]}
            Resume Data: {self.resume_data.dict()}
            
            Analyze and decide next action:
            1. If job field empty: ANALYZE_JOB
            2. If personal/professional info incomplete: INTERVIEW
            3. If all info collected: BUILD_RESUME
            
            Format: <THOUGHT>analysis</THOUGHT><ACTION>choice</ACTION>
            """,
            expected_output="Structured decision with reasoning",
            agent=self.supervisor()
        )

    @task
    def analyze_job(self, job_field: str) -> Task:
        return Task(
            description=f"""
            Analyze {job_field} position:
            1. Required skills/qualifications
            2. Industry trends
            3. Career paths
            4. Resume optimization tips
            
            Format: <ANALYSIS>insights</ANALYSIS>
            """,
            expected_output="Job field analysis",
            agent=self.job_analyzer()
        )

    @task
    def gather_info(self) -> Task:
        return Task(
            description=f"""
            Current info: {self.resume_data.dict()}
            
            Gather missing details about:
            1. Professional experience
            2. Education/certifications
            3. Skills/achievements
            
            Format: <QUESTIONS>follow-up questions</QUESTIONS>
            """,
            expected_output="Information gathering questions",
            agent=self.interviewer()
        )

    @task
    def build_resume(self) -> Task:
        return Task(
            description=f"""
            Create resume using: {self.resume_data.dict()}
            
            Include:
            1. Contact information
            2. Professional summary
            3. Experience
            4. Education
            5. Skills
            
            Format: <RESUME>html_content---FEEDBACK---feedback</RESUME>
            """,
            expected_output="HTML resume with feedback",
            agent=self.resume_builder()
        )

    @crew
    def resume_crew(self) -> Crew:
        return Crew(
            agents=[
                self.supervisor(),
                self.interviewer(),
                self.job_analyzer(),
                self.resume_builder()
            ],
            tasks=[
                self.analyze_situation(),
                self.analyze_job(""),  # Will be updated with actual job field
                self.gather_info(),
                self.build_resume()
            ],
            process=Process.sequential,
            verbose=True
        )

    def handle_input(self, user_input: str) -> Dict:
        """Process user input and return appropriate response"""
        self.add_to_history('user', user_input)
        
        # Update resume data based on input
        self._update_resume_data(user_input)
        
        # Get crew's response
        crew_response = self.resume_crew().kickoff()
        
        # Parse and validate response
        response = AgentResponse(
            type='message',
            content={'message': crew_response}
        )
        
        if '<RESUME>' in crew_response:
            resume, feedback = crew_response.split('---FEEDBACK---')
            response = AgentResponse(
                type='resume',
                content={
                    'resume': resume.strip(),
                    'feedback': feedback.strip()
                }
            )
        
        self.add_to_history('assistant', response.content.get('message', ''))
        return response.dict()

    def _update_resume_data(self, user_input: str) -> None:
        """Update resume data based on user input"""
        input_lower = user_input.lower()
        
        # Update job field
        if not self.resume_data.job_field and 'job' in input_lower:
            self.resume_data.job_field = user_input
            
        # Update personal info
        if '@' in input_lower:
            self.resume_data.personal_info.email = user_input
        # [Add other field updates]

if __name__ == "__main__":
    agent = ResumeCrewAI()
    result = agent.handle_input("What is your professional experience?")
    print(result) 