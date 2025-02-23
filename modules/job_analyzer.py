from crewai import Agent, Task

class JobAnalyzer:
    def __init__(self, llm):
        self.agent = Agent(
            role='Job Market Analyst',
            goal='Analyze job requirements and provide comprehensive insights',
            backstory="""I am a job market expert who deeply understands various industries
            and their requirements. I analyze job fields, identify key requirements, and provide
            strategic insights about career paths and industry trends. I help ensure resumes
            are tailored to specific roles and industries.""",
            llm=llm,
            verbose=True
        )

    def analyze_field(self, job_field: str) -> str:
        task = Task(
            description=f"""
            Analyze the {job_field} field and provide comprehensive insights...
            """,
            expected_output="Detailed job field analysis with structured insights",
            agent=self.agent
        )
        return task.execute_sync() 