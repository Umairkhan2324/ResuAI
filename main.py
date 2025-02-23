from crewai import Agent, Task, Crew, Process, LLM
from typing import Dict, List
from datetime import datetime
from config import GEMINI_API_KEY

def initialize_llm():
    return LLM(
        model="gemini/gemini-1.5-pro-latest",
        temperature=0.1,
        api_key=GEMINI_API_KEY
    )

class ResumeCrewAI:
    def __init__(self):
        self.llm = initialize_llm()
        self.conversation_history = []
        self.collected_info = {}
        
        # Initialize agents
        self.supervisor = Agent(
            role='Supervisor',
            goal='Coordinate resume creation process',
            backstory="""Expert career advisor coordinating resume creation through
            analysis and delegation to specialized agents.""",
            llm=self.llm,
            verbose=True
        )
        
        self.interviewer = Agent(
            role='Professional Interviewer',
            goal='Gather comprehensive professional information',
            backstory="""Expert interviewer gathering detailed career information.""",
            llm=self.llm,
            verbose=True
        )
        
        self.job_analyzer = Agent(
            role='Job Market Analyst',
            goal='Analyze job requirements and market trends',
            backstory="""Industry expert providing insights on requirements and trends.""",
            llm=self.llm,
            verbose=True
        )
        
        self.resume_builder = Agent(
            role='Resume Expert',
            goal='Create ATS-optimized resumes',
            backstory="""Expert resume writer creating compelling resumes.""",
            llm=self.llm,
            verbose=True
        )

    def add_to_history(self, role: str, message: str):
        self.conversation_history.append({
            'role': role,
            'message': message,
            'time': datetime.now()
        })

    def get_context(self) -> str:
        return "\n".join([
            f"{msg['role']}: {msg['message']}"
            for msg in self.conversation_history[-5:]
        ])

    def handle_input(self, user_input: str) -> Dict:
        self.add_to_history('user', user_input)
        
        # Update collected info
        input_lower = user_input.lower()
        if not self.collected_info.get('job_field') and 'job' in input_lower:
            self.collected_info['job_field'] = user_input
            
        # Create tasks based on current state
        tasks = []
        
        # Analysis task
        tasks.append(Task(
            description=f"""
            Analyze the current situation:
            Context: {self.get_context()}
            Info collected: {self.collected_info}
            
            Decide next action and respond appropriately.
            If job field is known but information is missing, ask relevant questions.
            If all information is collected, create the resume.
            
            Format: <THOUGHT>analysis</THOUGHT><ACTION>next_step</ACTION>
            """,
            expected_output="Analysis and next step",
            agent=self.supervisor
        ))
        
        # Create crew and execute
        crew = Crew(
            agents=[self.supervisor, self.interviewer, self.job_analyzer, self.resume_builder],
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
        
        response = crew.kickoff()
        
        # Handle resume creation if needed
        if '<RESUME>' in response:
            resume, feedback = response.split('---FEEDBACK---')
            return {
                'type': 'resume',
                'resume': resume.strip(),
                'feedback': feedback.strip()
            }
        
        # Regular response
        self.add_to_history('assistant', response)
        return {
            'type': 'message',
            'message': response
        }

if __name__ == "__main__":
    agent = ResumeCrewAI()
    result = agent.handle_input("What is your professional experience?")
    print(result) 