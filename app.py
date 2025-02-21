import streamlit as st
from main import initialize_llm, DynamicInterviewer, create_resume

def initialize_session_state():
    if 'stage' not in st.session_state:
        st.session_state.stage = 'start'
    if 'responses' not in st.session_state:
        st.session_state.responses = {}
    if 'current_question' not in st.session_state:
        st.session_state.current_question = None
    if 'interview_history' not in st.session_state:
        st.session_state.interview_history = []
    if 'field_analysis' not in st.session_state:
        st.session_state.field_analysis = {}
    if 'interviewer' not in st.session_state:
        st.session_state.interviewer = None

st.set_page_config(
    page_title="AI Resume Generator",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 ResuAI")
initialize_session_state()

# Initialize LLM
llm = initialize_llm()

def reset_session():
    for key in st.session_state.keys():
        del st.session_state[key]
    initialize_session_state()

def conduct_interview():
    # Display interview history
    for qa in st.session_state.interview_history:
        with st.chat_message("assistant"):
            st.write(qa["question"])
        with st.chat_message("user"):
            st.write(qa["answer"])

    # Get next question
    if st.session_state.current_question is None:
        st.session_state.current_question = st.session_state.interviewer.get_next_question(
            st.session_state.field_analysis,
            st.session_state.responses
        )

    # Display current question
    with st.chat_message("assistant"):
        st.write(st.session_state.current_question)

    # Get user response
    user_response = st.chat_input("Your answer:")
    if user_response:
        # Save response
        st.session_state.responses[st.session_state.current_question] = user_response
        st.session_state.interview_history.append({
            "question": st.session_state.current_question,
            "answer": user_response
        })
        
        # Check if interview is complete
        if st.session_state.interviewer.is_interview_complete(st.session_state.responses):
            st.session_state.stage = 'generate_resume'
        else:
            st.session_state.current_question = None
        st.rerun()

# Main application flow
if st.session_state.stage == 'start':
    st.markdown("""
    Welcome to ResuAI! Let's create your professional resume through an interactive interview.
    First, please enter your desired job field.
    """)
    
    job_field = st.text_input("Enter your desired job field:", placeholder="e.g., Software Engineering")
    
    if st.button("Start Interview", type="primary"):
        if job_field:
            st.session_state.field_analysis = {"field": job_field}
            st.session_state.interviewer = DynamicInterviewer(llm)
            st.session_state.stage = 'interview'
            st.rerun()
        else:
            st.warning("Please enter a job field first!")

elif st.session_state.stage == 'interview':
    conduct_interview()

elif st.session_state.stage == 'generate_resume':
    with st.spinner("Generating your resume..."):
        try:
            result = create_resume(st.session_state.interviewer.agent, st.session_state.responses)
            
            st.success("Resume generated successfully!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Generated Resume")
                st.markdown(result["resume"], unsafe_allow_html=True)
            
            with col2:
                st.subheader("AI Feedback")
                st.markdown(result["feedback"])
            
            if st.button("Start Over"):
                reset_session()
                st.rerun()
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Add helpful information at the bottom
st.markdown("---")
st.markdown("""
### How to use:
1. Enter your desired job field
2. Answer the interviewer's questions
3. Review your generated resume and feedback
4. Make adjustments as needed based on the AI feedback
""") 