from crewai import Agent, Task, Crew, Process, LLM
import google.generativeai as genai
from config import GEMINI_API_KEY
from modules.dynamic_interviewer import DynamicInterviewer
from modules.job_analyzer import JobAnalyzer
from modules.resume_builder import ResumeBuilder
from typing import Dict
from datetime import datetime

class GeminiLLM(LLM):
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-pro-latest')
        self.temperature = 0.1
        
    def call(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error calling Gemini: {str(e)}")
            return "I apologize, but I'm having trouble processing that request."

class SupervisorAgent:
    def __init__(self, llm):
        self.llm = llm
        self.conversation_history = []
        self.collected_info = {}
        
        # Initialize specialized agents
        self.interviewer = DynamicInterviewer(llm)
        self.analyzer = JobAnalyzer(llm)
        self.resume_builder = ResumeBuilder(llm)
        
        # Initialize supervisor as ReAct agent
        self.supervisor = Agent(
            role='Supervisor',
            goal='Coordinate resume creation process',
            backstory="""I am a senior career advisor who coordinates the resume creation process.
            I analyze situations, delegate tasks to specialized agents, and ensure all necessary
            information is collected through natural conversation.""",
            llm=llm,
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
            for msg in self.conversation_history[-5:]  # Keep last 5 messages for context
        ])

    def handle_input(self, user_input: str) -> Dict:
        self.add_to_history('user', user_input)
        
        # Let supervisor analyze and decide next action
        task = Task(
            description=f"""
            Analyze the current situation and decide next action:
            
            Conversation history:
            {self.get_context()}
            
            Collected information:
            {self.collected_info}
            
            Think through these steps:
            1. Analyze what information we have
            2. Determine what's missing
            3. Decide which specialist to engage
            
            Available actions:
            - ANALYZE_JOB: Get job field insights
            - INTERVIEW: Gather more information
            - BUILD_RESUME: Create final resume
            
            Format your response as:
            <THOUGHT>Your analysis of the situation</THOUGHT>
            <ACTION>ANALYZE_JOB|INTERVIEW|BUILD_RESUME</ACTION>
            <REASON>Why you chose this action</REASON>
            """,
            agent=self.supervisor
        )
        
        response = task.execute()
        action = self._extract_content(response, 'ACTION')
        
        # Execute the chosen action
        if action == 'ANALYZE_JOB':
            result = self.analyzer.analyze_field(user_input)
            self.collected_info['job_field'] = user_input
            return self._format_response('assistant', result)
            
        elif action == 'INTERVIEW':
            result = self.interviewer.gather_information(
                self.collected_info.get('job_field', ''),
                self.collected_info
            )
            return self._format_response('assistant', result)
            
        elif action == 'BUILD_RESUME':
            result = self.resume_builder.create_resume(
                self.collected_info.get('job_field', ''),
                self.collected_info
            )
            if '<RESUME>' in result:
                resume, feedback = result.split('---FEEDBACK---')
                return {
                    'type': 'resume',
                    'resume': resume.strip(),
                    'feedback': feedback.strip()
                }
        
        return self._format_response('assistant', response)

    def _extract_content(self, text: str, tag: str) -> str:
        start = text.find(f'<{tag}>') + len(tag) + 2
        end = text.find(f'</{tag}>')
        return text[start:end].strip() if start > -1 and end > -1 else ''

    def _format_response(self, role: str, message: str) -> Dict:
        self.add_to_history(role, message)
        return {
            'type': 'message',
            'message': message
        }

if __name__ == "__main__":
    llm = GeminiLLM()
    agent = SupervisorAgent(llm)
    result = agent.handle_input("What is your professional experience?")
    print(result) 