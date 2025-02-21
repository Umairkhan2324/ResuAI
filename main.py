from crewai import Agent, Task, Crew, Process, LLM
import google.generativeai as genai
from config import GEMINI_API_KEY
from typing import Dict, List
from datetime import datetime

def initialize_llm():
    genai.configure(api_key=GEMINI_API_KEY)
    return LLM(model="gemini/gemini-1.5-pro-latest", temperature=0.5)

class SupervisorAgent:
    def __init__(self, llm):
        self.llm = llm
        self.conversation_history = []
        self.collected_info = {}
        
        # Initialize all agents
        self.supervisor = Agent(
            role='Career Advisor Supervisor',
            goal='Coordinate the resume creation process intelligently',
            backstory="""I am a senior career advisor who manages the resume creation process.
            I analyze conversations, determine when to gather more information, and decide 
            when we have enough details to create a resume. I ensure the process is natural 
            and comprehensive.""",
            llm=self.llm,
            verbose=True
        )
        
        self.analyzer = Agent(
            role='Job Market Analyst',
            goal='Analyze job requirements and provide insights',
            backstory="""I deeply understand various job markets and their requirements.
            I can analyze job fields and provide valuable insights about required skills,
            qualifications, and industry trends.""",
            llm=self.llm,
            verbose=True
        )
        
        self.interviewer = Agent(
            role='Professional Interviewer',
            goal='Gather comprehensive professional information',
            backstory="""I am an expert interviewer who knows how to ask the right questions
            to gather relevant information for resumes. I adapt my questions based on previous
            responses and ensure all important details are covered.""",
            llm=self.llm,
            verbose=True
        )
        
        self.resume_builder = Agent(
            role='Resume Creator',
            goal='Create outstanding, ATS-friendly resumes',
            backstory="""I am an expert resume writer who knows how to present information
            effectively. I create compelling, ATS-optimized resumes that highlight candidates'
            strengths and achievements.""",
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
        
        # Let the supervisor analyze and decide
        task = Task(
            description=f"""
            As the supervisor, analyze the current conversation and decide the next step.
            
            Conversation history:
            {self.get_context()}
            
            Collected information:
            {self.collected_info}
            
            Your task is to:
            1. Understand the current state of the conversation
            2. Decide what information is still needed
            3. Choose which specialist (analyzer, interviewer, or resume builder) should handle this
            4. Provide clear instructions to the chosen specialist
            
            If this is a new conversation, start by welcoming the user and understanding their job field.
            If we're gathering information, ensure it's comprehensive before moving to resume creation.
            """,
            expected_output="Decision on next action and which specialist to involve",
            agent=self.supervisor
        )
        
        crew = Crew(
            agents=[self.supervisor],
            tasks=[task]
        )
        
        decision = crew.kickoff()
        
        # Based on the supervisor's decision, engage the appropriate specialist
        if "analyze" in decision.lower() or "job field" in decision.lower():
            return self._engage_analyzer(user_input)
        elif "interview" in decision.lower() or "question" in decision.lower():
            return self._engage_interviewer()
        elif "resume" in decision.lower() or "create" in decision.lower():
            return self._create_resume()
        else:
            # Default to supervisor's response
            return self._format_response('assistant', decision)

    def _engage_analyzer(self, job_field: str) -> Dict:
        task = Task(
            description=f"""
            Analyze the {job_field} field and provide insights.
            Consider:
            - Key requirements and qualifications
            - Essential skills (technical and soft)
            - Industry trends
            - Career progression paths
            
            Provide a helpful analysis that will guide our information gathering.
            """,
            expected_output="Comprehensive job field analysis",
            agent=self.analyzer
        )
        
        crew = Crew(agents=[self.analyzer], tasks=[task])
        analysis = crew.kickoff()
        
        self.collected_info['job_field'] = job_field
        self.collected_info['analysis'] = analysis
        return self._format_response('assistant', analysis)

    def _engage_interviewer(self) -> Dict:
        task = Task(
            description=f"""
            Based on the conversation history and collected information:
            {self.get_context()}
            {self.collected_info}
            
            Ask the next most relevant question to gather resume information.
            Consider what information is still missing and what would be most valuable to know.
            Make your question conversational and context-aware.
            """,
            expected_output="Next interview question",
            agent=self.interviewer
        )
        
        crew = Crew(agents=[self.interviewer], tasks=[task])
        question = crew.kickoff()
        return self._format_response('assistant', question)

    def _create_resume(self) -> Dict:
        task = Task(
            description=f"""
            Create a professional resume based on:
            {self.collected_info}
            
            1. Format it in clean HTML
            2. Ensure it's ATS-friendly
            3. Highlight key achievements and skills
            4. Add your expert feedback after a ---FEEDBACK--- marker
            """,
            expected_output="HTML resume with feedback",
            agent=self.resume_builder
        )
        
        crew = Crew(agents=[self.resume_builder], tasks=[task])
        result = crew.kickoff()
        
        try:
            resume, feedback = result.split('---FEEDBACK---')
        except:
            resume = result
            feedback = "Great resume! Consider adding more quantifiable achievements if possible."
            
        return {
            'type': 'resume',
            'resume': resume.strip(),
            'feedback': feedback.strip()
        }

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