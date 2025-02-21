from crewai import Agent, Task, Crew, Process, LLM
import google.generativeai as genai
from config import GEMINI_API_KEY
from typing import Dict, List
from datetime import datetime

def initialize_llm():
    genai.configure(api_key=GEMINI_API_KEY)
    return LLM(model="gemini/gemini-1.5-pro-latest", temperature=0.5)

class ResumeAgent:
    def __init__(self, llm):
        self.llm = llm
        self.conversation_history = []
        self.job_field = None
        
        # Create our core agent
        self.agent = Agent(
            role='Career Advisor',
            goal='Help create professional resumes through conversation',
            backstory="""I am an expert career advisor who helps people create 
            professional resumes. I gather information through conversation and 
            provide tailored guidance.""",
            llm=llm,
            verbose=True
        )

    def add_to_history(self, role: str, message: str):
        self.conversation_history.append({
            'role': role,
            'message': message,
            'time': datetime.now()
        })

    def get_conversation_context(self) -> str:
        return "\n".join([
            f"{msg['role']}: {msg['message']}" 
            for msg in self.conversation_history[-5:]  # Last 5 messages for context
        ])

    def handle_input(self, user_input: str) -> Dict:
        # Add user input to history
        self.add_to_history('user', user_input)

        # If this is the first input, set it as job field
        if not self.job_field:
            self.job_field = user_input
            return self.analyze_job_field()

        # Get next response based on conversation history
        task = Task(
            description=f"""
            Based on the conversation about a {self.job_field} position, determine the best response.
            Previous conversation:
            {self.get_conversation_context()}
            
            If we have enough information, create the resume. Otherwise, ask relevant questions.
            Focus on gathering:
            1. Professional experience
            2. Education
            3. Skills
            4. Achievements
            5. Career goals
            """,
            expected_output="Either a next question or a complete resume with feedback",
            agent=self.agent
        )

        crew = Crew(
            agents=[self.agent],
            tasks=[task]
        )

        response = crew.kickoff()
        self.add_to_history('assistant', response)

        # Check if response contains a resume
        if "<html" in response:
            return {
                'action': 'complete',
                'resume': self.extract_resume(response),
                'feedback': self.extract_feedback(response)
            }
        else:
            return {
                'action': 'continue',
                'message': response
            }

    def analyze_job_field(self) -> Dict:
        task = Task(
            description=f"""
            Analyze the {self.job_field} field and provide initial guidance.
            Include:
            1. Key requirements
            2. Important skills
            3. Industry trends
            4. What information we should gather from the candidate
            """,
            expected_output="A helpful analysis and initial question",
            agent=self.agent
        )

        crew = Crew(
            agents=[self.agent],
            tasks=[task]
        )

        response = crew.kickoff()
        self.add_to_history('assistant', response)
        
        return {
            'action': 'continue',
            'message': response
        }

    def extract_resume(self, response: str) -> str:
        try:
            start = response.find("<html")
            end = response.find("</html>") + 7
            return response[start:end]
        except:
            return f"<html><body><p>{response}</p></body></html>"

    def extract_feedback(self, response: str) -> str:
        try:
            end = response.find("<html")
            if end == -1:
                return response
            return response[:end].strip()
        except:
            return "No specific feedback available."

if __name__ == "__main__":
    llm = initialize_llm()
    agent = ResumeAgent(llm)
    result = agent.handle_input("What is your professional experience?")
    print(result) 