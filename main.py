from crewai import Agent, Task, Crew, Process, LLM
import google.generativeai as genai
from config import GEMINI_API_KEY
from typing import Dict
from datetime import datetime

def initialize_llm():
    genai.configure(api_key=GEMINI_API_KEY)
    return LLM(model="gemini/gemini-1.5-pro-latest", temperature=0.5)

class SupervisorAgent:
    def __init__(self, llm):
        self.llm = llm
        self.conversation_history = []
        self.collected_info = {}
        
        self.supervisor = Agent(
            role='Career Advisor',
            goal='Guide resume creation through natural conversation',
            backstory="""I am an expert career advisor who helps create professional resumes. 
            I understand various job markets, know what information to gather, and can create 
            compelling resumes. I maintain natural conversations while ensuring all necessary 
            information is collected.""",
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
            for msg in self.conversation_history[-5:]  # Keep last 5 messages for context
        ])

    def handle_input(self, user_input: str) -> Dict:
        self.add_to_history('user', user_input)
        
        task = Task(
            description=f"""
            You are a career advisor helping create a professional resume.
            
            Current conversation:
            {self.get_context()}
            
            Information collected so far:
            {self.collected_info}
            
            Your task:
            1. If this is a new conversation:
               - Warmly greet the user
               - Ask about their desired job field
            
            2. If the user mentions a job field for the first time:
               - Acknowledge their choice
               - Provide brief insights about the field
               - Ask about their specific interests within that field
            
            3. Once you know their specific interests:
               - Ask focused questions about:
                 * Professional experience in that area
                 * Relevant education and certifications
                 * Key skills and competencies
                 * Notable achievements
                 * Projects or initiatives
               - Keep questions conversational and relevant to their field
               - Ask one or two questions at a time to maintain flow
            
            4. Only create a resume when you have gathered comprehensive information about:
               - Their specific role preference
               - Professional experience
               - Education
               - Skills
               - Achievements
            
            When creating the resume, format it as:
            <RESUME>
            [HTML resume content]
            ---FEEDBACK---
            [Your feedback]
            </RESUME>
            
            Otherwise, continue the natural conversation and gather information.
            Remember to:
            - Keep the conversation friendly and professional
            - Ask follow-up questions based on their responses
            - Show understanding of their field and career goals
            - Store important information they share
            """,
            expected_output="Either a conversational response or a complete resume with feedback",
            agent=self.supervisor
        )
        
        crew = Crew(
            agents=[self.supervisor],
            tasks=[task]
        )
        
        response = str(crew.kickoff())
        
        # Check if response contains a resume
        if "<RESUME>" in response:
            try:
                resume_content = response.split("<RESUME>")[1].split("</RESUME>")[0].strip()
                resume, feedback = resume_content.split("---FEEDBACK---")
                return {
                    'type': 'resume',
                    'resume': resume.strip(),
                    'feedback': feedback.strip()
                }
            except:
                return self._format_response('assistant', response)
        else:
            # Update collected info based on response content
            if "job field" in response.lower() and not self.collected_info.get('job_field'):
                self.collected_info['job_field'] = user_input
            
            # Extract and store other information based on context
            if "experience" in user_input.lower():
                self.collected_info['experience'] = user_input
            if "education" in user_input.lower():
                self.collected_info['education'] = user_input
            if "skills" in user_input.lower():
                self.collected_info['skills'] = user_input
            if "achievements" in user_input.lower():
                self.collected_info['achievements'] = user_input
            
            return self._format_response('assistant', response)

    def _format_response(self, role: str, message: str) -> Dict:
        self.add_to_history(role, message)
        return {
            'type': 'message',
            'message': message
        }

if __name__ == "__main__":
    llm = initialize_llm()
    agent = SupervisorAgent(llm)
    result = agent.handle_input("What is your professional experience?")
    print(result) 