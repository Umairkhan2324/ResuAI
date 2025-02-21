from crewai import Agent, Task, Crew, Process, LLM
import google.generativeai as genai
from config import GEMINI_API_KEY
from typing import Dict, List
from datetime import datetime
import json

def initialize_llm():
    genai.configure(api_key=GEMINI_API_KEY)
    return LLM(model="gemini/gemini-1.5-pro-latest", temperature=0.5)

class SupervisorAgent:
    def __init__(self, llm):
        self.llm = llm
        self.conversation_history = []
        self.collected_info = {}
        self.current_stage = 'greeting'
        
        # Initialize all agents
        self.supervisor = self._create_supervisor()
        self.analyzer = self._create_analyzer()
        self.interviewer = self._create_interviewer()
        self.resume_builder = self._create_resume_builder()

    def _create_supervisor(self):
        return Agent(
            role='Career Advisor Supervisor',
            goal='Coordinate the resume creation process and ensure quality',
            backstory="""I am a senior career advisor who manages the resume creation process.
            I understand when to analyze job requirements, when to gather information, and when
            to create the final resume. I ensure all necessary information is collected.""",
            llm=self.llm,
            verbose=True
        )

    def _create_analyzer(self):
        return Agent(
            role='Job Market Analyst',
            goal='Analyze job fields and requirements',
            backstory='Expert in analyzing job markets and understanding requirements',
            llm=self.llm,
            verbose=True
        )

    def _create_interviewer(self):
        return Agent(
            role='Professional Interviewer',
            goal='Gather detailed professional information',
            backstory='Expert at conducting professional interviews and gathering relevant information',
            llm=self.llm,
            verbose=True
        )

    def _create_resume_builder(self):
        return Agent(
            role='Resume Creator',
            goal='Create professional, ATS-friendly resumes',
            backstory='Expert resume writer who creates compelling resumes',
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
        
        # Let supervisor decide next action
        task = Task(
            description=f"""
            Current stage: {self.current_stage}
            Previous conversation: {self.get_context()}
            Collected information: {json.dumps(self.collected_info, indent=2)}

            Determine the next best action:
            1. If greeting, welcome user and ask for job field
            2. If new job field mentioned, analyze requirements
            3. If gathering information, ask relevant questions
            4. If enough information, create resume

            Respond in JSON format:
            {{
                "action": "greet" | "analyze" | "interview" | "create_resume",
                "agent": "supervisor" | "analyzer" | "interviewer" | "resume_builder",
                "message": "message to user",
                "next_stage": "next stage name"
            }}
            """,
            expected_output="JSON response with next action",
            agent=self.supervisor
        )

        crew = Crew(
            agents=[self.supervisor],
            tasks=[task]
        )

        try:
            response = json.loads(crew.kickoff())
            self.current_stage = response['next_stage']
            
            if response['action'] == 'greet':
                return self._format_response('assistant', response['message'])
            
            elif response['action'] == 'analyze':
                analysis = self._analyze_job_field(user_input)
                self.collected_info['job_field'] = user_input
                self.collected_info['analysis'] = analysis
                return self._format_response('assistant', analysis)
            
            elif response['action'] == 'interview':
                question = self._get_next_question()
                return self._format_response('assistant', question)
            
            elif response['action'] == 'create_resume':
                result = self._create_resume()
                return {
                    'type': 'resume',
                    'resume': result['resume'],
                    'feedback': result['feedback']
                }
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return self._format_response('assistant', 
                "I apologize, but I'm having trouble processing that. Could you please rephrase?")

    def _format_response(self, role: str, message: str) -> Dict:
        self.add_to_history(role, message)
        return {
            'type': 'message',
            'message': message
        }

    def _analyze_job_field(self, job_field: str) -> str:
        task = Task(
            description=f"Analyze the {job_field} field requirements and expectations",
            expected_output="Detailed analysis and initial question",
            agent=self.analyzer
        )
        
        crew = Crew(
            agents=[self.analyzer],
            tasks=[task]
        )
        
        return crew.kickoff()

    def _get_next_question(self) -> str:
        task = Task(
            description=f"""
            Based on collected information: {json.dumps(self.collected_info, indent=2)}
            Previous conversation: {self.get_context()}
            
            Ask the next most relevant question to gather resume information.
            """,
            expected_output="Next interview question",
            agent=self.interviewer
        )
        
        crew = Crew(
            agents=[self.interviewer],
            tasks=[task]
        )
        
        return crew.kickoff()

    def _create_resume(self) -> Dict:
        task = Task(
            description=f"""
            Create a professional resume using:
            Collected information: {json.dumps(self.collected_info, indent=2)}
            
            Provide both HTML resume and feedback.
            Separate resume and feedback with ---FEEDBACK--- marker.
            """,
            expected_output="HTML resume and feedback",
            agent=self.resume_builder
        )
        
        crew = Crew(
            agents=[self.resume_builder],
            tasks=[task]
        )
        
        response = crew.kickoff()
        parts = response.split('---FEEDBACK---')
        
        return {
            'resume': parts[0].strip(),
            'feedback': parts[1].strip() if len(parts) > 1 else "No feedback provided"
        }

if __name__ == "__main__":
    llm = initialize_llm()
    agent = SupervisorAgent(llm)
    result = agent.handle_input("What is your professional experience?")
    print(result) 