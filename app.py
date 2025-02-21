import streamlit as st
from main import initialize_llm, SupervisorAgent

def initialize_session_state():
    if 'agent' not in st.session_state:
        llm = initialize_llm()
        st.session_state.agent = SupervisorAgent(llm)

st.set_page_config(
    page_title="AI Resume Generator",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 ResuAI")
initialize_session_state()

# Display conversation history
for msg in st.session_state.agent.conversation_history:
    with st.chat_message(msg['role']):
        st.write(msg['message'])

# Show initial message if conversation hasn't started
if not st.session_state.agent.conversation_history:
    st.markdown("""
    Welcome to ResuAI! Let's create your professional resume through a conversation.
    Please start by telling me what field you're interested in (e.g., "Software Engineering").
    """)

# Get user input
user_input = st.chat_input("Your response:")
if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    
    with st.spinner("Thinking..."):
        result = st.session_state.agent.handle_input(user_input)
        
        if result['type'] == 'resume':
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Your Resume")
                st.markdown(result['resume'], unsafe_allow_html=True)
            with col2:
                st.subheader("Feedback")
                st.markdown(result['feedback'])
            
            if st.button("Start Over"):
                st.session_state.agent = SupervisorAgent(initialize_llm())
                st.rerun()
        else:
            with st.chat_message("assistant"):
                st.write(result['message'])

# Add helpful information at the bottom
st.markdown("---")
st.markdown("""
### How to use:
1. Enter your desired job field
2. Answer the questions about your experience and skills
3. Review your generated resume and feedback
4. Start over if you want to make changes
""") 