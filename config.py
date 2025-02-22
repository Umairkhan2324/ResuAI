import os
from dotenv import load_dotenv

load_dotenv()

# IBM WatsonX credentials
IBM_API_KEY = os.getenv('IBM_API_KEY')
IBM_PROJECT_ID = os.getenv('IBM_PROJECT_ID')
IBM_URL = os.getenv('IBM_URL', 'https://us-south.ml.cloud.ibm.com') 