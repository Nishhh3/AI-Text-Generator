"""
AI Text Generator with Sentiment Analysis
A Streamlit application that generates text based on sentiment analysis.
"""

import streamlit as st
from models import SentimentTextGenerator
import time

# Page configuration
st.set_page_config(
    page_title="AI Text Generator",
    page_icon="✨",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #6c757d;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize the model (cached to avoid reloading)
@st.cache_resource
def load_models():
    """Load and cache the models."""
    return SentimentTextGenerator()

def main():
    # Header
    st.markdown('<div class="main-header">✨ AI Text Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Generate sentiment-aligned text using AI</div>', unsafe_allow_html=True)
    
    # Load models
    try:
        with st.spinner("Loading AI models... (This may take a minute on first run)"):
            model = load_models()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()
    
    # Sidebar for settings
    st.sidebar.title("⚙️ Settings")
    
    # Mode selection
    mode = st.sidebar.radio(
        "Choose Mode:",
        ["Auto-detect Sentiment", "Manual Sentiment Selection"],
        help="Auto-detect analyzes your prompt, Manual lets you choose the sentiment"
    )
    
    # Manual sentiment selection
    manual_sentiment = None
    if mode == "Manual Sentiment Selection":
        manual_sentiment = st.sidebar.selectbox(
            "Select Sentiment:",
            ["positive", "negative", "neutral"]
        )
    
    # Text generation parameters
    st.sidebar.subheader("Generation Parameters")
    max_length = st.sidebar.slider(
        "Maximum Length (words)",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="Controls the length of generated text"
    )
    
    temperature = st.sidebar.slider(
        "Creativity (Temperature)",
        min_value=0.5,
        max_value=1.5,
        value=0.8,
        step=0.1,
        help="Higher values make output more creative but less focused"
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Input Prompt")
        user_prompt = st.text_area(
            "Enter your prompt:",
            height=150,
            placeholder="E.g., 'Write about a sunny day at the beach' or 'Describe a disappointing experience'",
            help="Enter a prompt for the AI to generate text from"
        )
        
        generate_button = st.button("🚀 Generate Text", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("📄 Generated Output")
        output_container = st.container()
    
    # Generate text when button is clicked
    if generate_button:
        if not user_prompt.strip():
            st.warning("⚠️ Please enter a prompt first!")
        else:
            with st.spinner("🤖 Analyzing sentiment and generating text..."):
                try:
                    if mode == "Auto-detect Sentiment":
                        # Auto-detect sentiment
                        generated_text, sentiment_info = model.generate_with_auto_sentiment(
                            user_prompt,
                            max_length=max_length,
                            temperature=temperature
                        )
                        detected_sentiment = sentiment_info['sentiment']
                        confidence = sentiment_info['confidence']
                        
                        # Display sentiment analysis
                        with output_container:
                            st.info(f"**Detected Sentiment:** {detected_sentiment.capitalize()} (Confidence: {confidence:.2%})")
                    else:
                        # Manual sentiment
                        generated_text = model.generate_text(
                            user_prompt,
                            sentiment=manual_sentiment,
                            max_length=max_length,
                            temperature=temperature
                        )
                        detected_sentiment = manual_sentiment
                        
                        # Display selected sentiment
                        with output_container:
                            st.info(f"**Selected Sentiment:** {detected_sentiment.capitalize()}")
                    
                    # Display generated text
                    with output_container:
                        st.markdown("**Generated Text:**")
                        
                        # Color code based on sentiment
                        sentiment_class = f"sentiment-{detected_sentiment}"
                        st.markdown(f'<div class="{sentiment_class}">', unsafe_allow_html=True)
                        st.write(generated_text)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Word count
                        word_count = len(generated_text.split())
                        st.caption(f"📊 Word count: {word_count}")
                        
                        # Download button
                        st.download_button(
                            label="⬇️ Download Text",
                            data=generated_text,
                            file_name="generated_text.txt",
                            mime="text/plain"
                        )
                
                except Exception as e:
                    st.error(f"❌ Error generating text: {str(e)}")
                    st.info("Try adjusting the parameters or simplifying your prompt.")
    
    # Information section
    with st.expander("ℹ️ About This Application"):
        st.markdown("""
        ### How it works:
        
        1. **Sentiment Analysis**: The system analyzes your input prompt to determine its sentiment (positive, negative, or neutral)
        2. **Text Generation**: Based on the detected sentiment, the AI generates coherent text that aligns with that emotional tone
        3. **Customization**: You can adjust the length and creativity of the generated text
        
        ### Models Used:
        - **Sentiment Analysis**: DistilBERT fine-tuned on SST-2 dataset
        - **Text Generation**: DistilGPT-2 (lightweight GPT-2 variant)
        
        ### Tips for Best Results:
        - Be specific in your prompts
        - Use descriptive language
        - Adjust temperature for more/less creative output
        - Try both auto-detect and manual sentiment modes
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>Built with Streamlit & Hugging Face Transformers</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()