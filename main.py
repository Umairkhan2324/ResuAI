from crewai import Agent, Task, Crew, Process, LLM
from langchain.memory import ConversationBufferMemory
import google.generativeai as genai
from config import GEMINI_API_KEY
from typing import Dict
from datetime import datetime

class GeminiLLM(LLM):
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-pro-latest')
        self.stop = []  # Add stop sequences
        self.temperature = 0.1
        self.max_tokens = 1024
        
    def call(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': self.temperature,
                    'max_output_tokens': self.max_tokens,
                    'stop_sequences': self.stop
                }
            )
            return response.text
        except Exception as e:
            print(f"Error calling Gemini: {str(e)}")
            return "I apologize, but I'm having trouble processing that request."
            
    def __getattr__(self, name):
        # Handle any missing attributes that CrewAI might expect
        if name == 'stop':
            return []
        return None

def initialize_llm():
    return GeminiLLM()

class SupervisorAgent:
    def __init__(self, llm):
        self.llm = llm
        self.conversation_history = []
        self.collected_info = {}
        self.current_stage = 'greeting'
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.supervisor = Agent(
            role='Career Advisor',
            goal='Guide resume creation through natural conversation',
            backstory="""I am an expert career advisor who helps create professional resumes. 
            I analyze responses, track collected information, and make informed decisions about 
            what information is still needed. I ensure a natural conversation flow while 
            systematically gathering all required details.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            memory=self.memory,
            tools=[],  # Add empty tools list
            max_iterations=3,  # Add max iterations
            max_rpm=10  # Add rate limit
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
        self._update_collected_info(user_input)
        
        task = Task(
            description=f"""
            You are a career advisor helping create a professional resume.
            
            Current conversation:
            {self.get_context()}
            
            Information collected so far:
            {self.collected_info}
            
            Current stage: {self.current_stage}
            
            Think through this step by step:
            1. Analyze what information we have
            2. Determine what's still missing
            3. Decide the next best action
            
            Stage Progression:
            - greeting → job_field → specific_interests → personal_info → professional_info → resume
            
            Required Information:
            1. Job Field Info (Status: {'job_field' in self.collected_info})
               - Target field
               - Specific interests
            
            2. Personal Info (Status: {all(key in self.collected_info for key in ['name', 'email', 'location', 'phone'])})
               - Full name
               - Email
               - Phone
               - Location
            
            3. Professional Info (Status: {all(key in self.collected_info for key in ['experience', 'education', 'skills'])})
               - Experience
               - Education
               - Skills
               - Achievements
            
            If ALL information is collected, create the resume.
            If ANY information is missing, ask for it naturally.
            
            Format your response as:
            <THOUGHT>Your analysis of the situation</THOUGHT>
            <ACTION>next_action: [ask_job_field|gather_personal|gather_professional|create_resume]</ACTION>
            <RESPONSE>Your actual response to the user</RESPONSE>
            """,
            expected_output="Structured response with thought process and action",
            agent=self.supervisor
        )
        
        crew = Crew(
            agents=[self.supervisor],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )
        
        response = str(crew.kickoff())
        
        # Parse the ReAct response
        thought = self._extract_content(response, 'THOUGHT')
        action = self._extract_content(response, 'ACTION')
        message = self._extract_content(response, 'RESPONSE')
        
        # Update stage based on action
        if 'ask_job_field' in action:
            self.current_stage = 'job_field'
        elif 'gather_personal' in action:
            self.current_stage = 'personal_info'
        elif 'gather_professional' in action:
            self.current_stage = 'professional_info'
        elif 'create_resume' in action:
            self.current_stage = 'resume'
            
            if '<RESUME>' in message:
                try:
                    resume_content = message.split("<RESUME>")[1].split("</RESUME>")[0].strip()
                    resume, feedback = resume_content.split("---FEEDBACK---")
                    return {
                        'type': 'resume',
                        'resume': resume.strip(),
                        'feedback': feedback.strip()
                    }
                except:
                    return self._format_response('assistant', message)
        
        return self._format_response('assistant', message)

    def _extract_content(self, text: str, tag: str) -> str:
        start = text.find(f'<{tag}>') + len(tag) + 2
        end = text.find(f'</{tag}>')
        return text[start:end].strip() if start > -1 and end > -1 else ''

    def _update_collected_info(self, user_input: str):
        # More sophisticated information extraction
        input_lower = user_input.lower()
        
        # Update job field info
        if self.current_stage == 'job_field' and not self.collected_info.get('job_field'):
            self.collected_info['job_field'] = user_input
            
        # Update personal info
        if any(word in input_lower for word in ['name', 'called']):
            self.collected_info['name'] = user_input
        if '@' in input_lower:
            self.collected_info['email'] = user_input
        if any(word in input_lower for word in ['live', 'located', 'city', 'state']):
            self.collected_info['location'] = user_input
        if any(word in input_lower for word in ['phone', 'contact', 'number']):
            self.collected_info['phone'] = user_input
            
        # Update professional info
        if 'experience' in input_lower:
            self.collected_info['experience'] = user_input
        if 'education' in input_lower:
            self.collected_info['education'] = user_input
        if 'skill' in input_lower:
            self.collected_info['skills'] = user_input
        if any(word in input_lower for word in ['achieve', 'accomplish']):
            self.collected_info['achievements'] = user_input

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