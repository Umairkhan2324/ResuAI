from crewai import Agent, Task

class FeedbackAgent:
    def __init__(self, llm):
        self.agent = Agent(
            role='Resume Review Specialist',
            goal='Provide actionable feedback on resumes',
            backstory='Professional resume reviewer with expertise in multiple industries',
            llm=llm,
            verbose=True
        )

    def get_feedback(self, resume: str) -> str:
        task = Task(
            description="Review the resume and provide detailed, actionable feedback",
            expected_output="Comprehensive feedback with specific improvement suggestions",
            agent=self.agent
        )
        
        return task.execute() 