import streamlit as st
from main import initialize_llm, SupervisorAgent

def initialize_session_state():
    if 'supervisor' not in st.session_state:
        st.session_state.supervisor = None
    if 'conversation_started' not in st.session_state:
        st.session_state.conversation_started = False

st.set_page_config(
    page_title="AI Resume Generator",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 ResuAI")
initialize_session_state()

# Initialize supervisor if needed
if not st.session_state.supervisor:
    llm = initialize_llm()
    st.session_state.supervisor = SupervisorAgent(llm)

# Display conversation history
for interaction in st.session_state.supervisor.memory.history:
    with st.chat_message(interaction['role']):
        st.write(interaction['message'])

# Get user input
if not st.session_state.conversation_started:
    st.markdown("""
    Welcome to ResuAI! Let's create your professional resume through an interactive conversation.
    Please enter your desired job field to begin.
    """)
    st.session_state.conversation_started = True

user_input = st.chat_input("Your response:")
if user_input:
    # Show user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Get and show AI response
    with st.spinner("Thinking..."):
        result = st.session_state.supervisor.handle_input(user_input)
        
        if result['action'] == 'generate_resume':
            # Display the final resume and feedback
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Generated Resume")
                st.markdown(result['response']['resume'], unsafe_allow_html=True)
            with col2:
                st.subheader("AI Feedback")
                st.markdown(result['response']['feedback'])
                
            if st.button("Start Over"):
                st.session_state.supervisor = None
                st.session_state.conversation_started = False
                st.rerun()
        else:
            # Show the next question or response
            with st.chat_message("assistant"):
                st.write(result['response'])

# Add helpful information at the bottom
st.markdown("---")
st.markdown("""
### How to use:
1. Enter your desired job field
2. Answer the interviewer's questions
3. Review your generated resume and feedback
4. Make adjustments as needed based on the AI feedback
""") 