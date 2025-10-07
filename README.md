# AI Text Generator
This project is an AI-powered text generator that analyzes the sentiment (positive, negative, or neutral) of a user's input and generates a new paragraph aligned with that sentiment.
The frontend is built using Streamlit, offering a simple interface for entering prompts and viewing generated text.

## Setup Guide
1. Clone the Repository
```
git clone https://github.com/Nishhh3/AI-Text-Generator.git
cd AI-Text-Generator
```
2. Create a env (or use a existing miniconda env)
```
python -m venv venv
venv\Scripts\activate # On Windows
or
source venv/bin/activate # On macOS/Linux
or
FOR MINICONDA
conda activate your_env_name
```
3. Install Dependencies
```
pip install -r requirements.txt
```
4. Run the models.py
```
python models.py
```
5. Run the App Locally
```
streamlit run app.py
```
The app will open automatically in your web browser

## Features
- Detects sentiment of input text automatically
- Generates text aligned with that sentiment
- Option to manually select sentiment
- Adjustable text length
- Built with Streamlit for easy interaction
