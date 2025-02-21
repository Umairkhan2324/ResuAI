import streamlit as st
from main import run_crew, ResumeData

st.set_page_config(
    page_title="AI Resume Generator",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Resume Generator")

st.markdown("""
This application uses CrewAI with Gemini to generate professional resumes.
The system employs four AI agents:
- A Job Field Analyzer who understands industry requirements
- A Professional Interviewer who gathers your information
- A Resume Builder who creates your resume
- A Feedback Agent who provides improvement suggestions
""")

# Input section
job_field = st.text_input("Enter your desired job field:", placeholder="e.g., Software Engineering")

if st.button("Generate Resume", type="primary"):
    if job_field:
        with st.spinner("Working on your resume..."):
            try:
                result = run_crew(job_field)
                
                # Display results
                st.success("Resume generated successfully!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Generated Resume")
                    # Ensure the HTML is properly rendered
                    if isinstance(result, dict) and 'resume' in result:
                        st.markdown(result['resume'], unsafe_allow_html=True)
                    else:
                        st.error("Resume generation failed")
                
                with col2:
                    st.subheader("AI Feedback")
                    if isinstance(result, dict) and 'feedback' in result:
                        st.markdown(result['feedback'])
                    else:
                        st.error("Feedback generation failed")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a job field first!")

# Add some helpful information at the bottom
st.markdown("---")
st.markdown("""
### How to use:
1. Enter your desired job field
2. Click 'Generate Resume'
3. Review the generated resume and feedback
4. Make any necessary adjustments based on the AI feedback
""") 