# ResuAI

An intelligent resume creation assistant powered by Google's Gemini Pro and CrewAI, providing a conversational interface for generating professional resumes.

## Features

- 🤖 Conversational AI Interface
- 📝 Dynamic Resume Generation
- 💡 Personalized Feedback
- 🎯 Field-Specific Guidance
- 🔄 Natural Conversation Flow
- 📊 Structured Information Gathering
- 🧠 ReAct-based Decision Making

## Technical Stack

- **Frontend**: Streamlit
- **AI Model**: Google Gemini Pro 1.5
- **Framework**: CrewAI
- **Language**: Python 3.11+

## Architecture

### Core Components

1. **SupervisorAgent**
   - ReAct-based decision making
   - Stage-based conversation management
   - Systematic information collection
   - Intelligent state tracking
   - Context-aware responses

2. **LLM Integration**
   - Google Gemini Pro 1.5
   - Temperature: 0.1 for consistent responses
   - Structured output parsing
   - Error handling and recovery

3. **Information Management**
   - Stage-based progression
   - Comprehensive data validation
   - Context-aware updates
   - Progressive information gathering

## Setup

1. **Environment Variables**
```env
IBM_API_KEY=your_api_key
IBM_PROJECT_ID=your_project_id
IBM_URL=your_watson_url
```

2. **Dependencies**
```bash
pip install -r requirements.txt
```

3. **Run Application**
```bash
streamlit run app.py
```

## Information Collection Flow

1. **Initial Contact**
   - Greeting
   - Job field identification
   - Field-specific insights

2. **Personal Information**
   - Full name
   - Professional email
   - Phone number
   - Location
   - Professional profiles

3. **Professional Background**
   - Work experience
   - Education
   - Technical skills
   - Achievements
   - Projects

## Resume Generation

- **Format**: HTML with structured sections
- **Sections**:
  - Contact Information
  - Professional Summary
  - Experience
  - Education
  - Skills
  - Projects
  - Achievements

## Technical Details

### AI Model Configuration
```python
generate_params = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MAX_NEW_TOKENS: 1024,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.TEMPERATURE: 0.5,
    GenParams.REPETITION_PENALTY: 1.0,
}
```

### Conversation Management
- Maintains 5-message context window
- Structured information storage
- Progressive data collection
- Validation before resume generation

## Usage

1. Start the application
2. Enter your target job field
3. Follow the conversational prompts
4. Provide personal and professional information
5. Review and receive your generated resume

## Error Handling

- API connection failures
- Incomplete information detection
- Conversation recovery
- Data validation

## Future Improvements

- Multiple resume templates
- PDF export functionality
- ATS optimization
- Direct job application integration
- Resume comparison tools

## Contributing

Feel free to submit issues and enhancement requests!

## License

[Your License Choice] 
