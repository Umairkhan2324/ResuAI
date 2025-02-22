from crewai import Agent, Task, Crew, Process, LLM
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watsonx_ai.foundation_models.extensions.langchain import WatsonxLLM
from config import IBM_API_KEY, IBM_PROJECT_ID, IBM_URL
from typing import Dict
from datetime import datetime

def initialize_llm():
    # Initialize WatsonX model parameters
    generate_params = {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.MAX_NEW_TOKENS: 1024,
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.TEMPERATURE: 0.5,
        GenParams.REPETITION_PENALTY: 1.0,
    }

    # Initialize credentials
    # credentials = Credentials(
    #     api_key=IBM_API_KEY,
    #     url=IBM_URL
    # )

    # Initialize the model
    model = Model(
        model_id="ibm/granite-13b-instruct-v1",
        params=generate_params,
        credentials={
            "apikey": IBM_API_KEY,
            "url": IBM_URL
        },
        project_id=IBM_PROJECT_ID
    )

    # Create custom LLM class for CrewAI compatibility
    class WatsonXLLM(LLM):
        def __init__(self, model):
            self.model = model
            
        def call(self, prompt: str) -> str:
            try:
                formatted_prompt = f"<instruction>{prompt}</instruction>"
                response = self.model.generate(formatted_prompt)
                return response.generated_text
            except Exception as e:
                print(f"Error calling WatsonX: {str(e)}")
                return "I apologize, but I'm having trouble processing that request."

    return WatsonXLLM(model)

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
            
            3. Once you know their specific interests, gather information in this order:
               a. Personal Information (if not collected):
                  - Full name
                  - Professional email
                  - Phone number
                  - Location (City, State)
                  - LinkedIn profile (if available)
                  - Portfolio/Github (if relevant)
               
               b. Professional Background:
                  - Professional experience
                  - Education and certifications
                  - Technical skills
                  - Notable achievements
                  - Projects and contributions
            
               Keep questions conversational and ask 1-2 questions at a time.
               Ensure personal information is collected before moving to resume creation.
            
            4. Only create a resume when you have gathered comprehensive information about:
               - Personal details (name, contact info, location)
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
            - Ensure all necessary personal information is collected
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
            # Verify we have personal information before creating resume
            if not all(key in self.collected_info for key in ['name', 'email', 'location']):
                return self._format_response('assistant', 
                    "Before I create your resume, I'll need some personal information. "
                    "Could you please share your full name, professional email, and location (city, state)?")
            
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
            # Update collected info based on context
            if "name" in user_input.lower():
                self.collected_info['name'] = user_input
            if "email" in user_input.lower():
                self.collected_info['email'] = user_input
            if "location" in user_input.lower() or "city" in user_input.lower():
                self.collected_info['location'] = user_input
            if "phone" in user_input.lower():
                self.collected_info['phone'] = user_input
            if "linkedin" in user_input.lower():
                self.collected_info['linkedin'] = user_input
            if "github" in user_input.lower() or "portfolio" in user_input.lower():
                self.collected_info['portfolio'] = user_input
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