from crewai import Agent

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
        return self.agent.execute_task(
            f"""
            Analyze the {job_field} field and provide comprehensive insights:

            1. Role Analysis:
               - Core responsibilities
               - Required technical skills
               - Essential soft skills
               - Experience level expectations

            2. Industry Context:
               - Current market trends
               - Growth opportunities
               - Key industry challenges
               - Technology requirements

            3. Career Development:
               - Typical career progression
               - Valuable certifications
               - Learning priorities
               - Growth opportunities

            4. Resume Focus Points:
               - Key achievements to highlight
               - Important keywords for ATS
               - Critical qualifications
               - Preferred experience format

            Format response as:
            <ANALYSIS>
            [Your detailed analysis structured under the above categories]
            </ANALYSIS>
            """
        ) 