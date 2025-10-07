from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForCausalLM,
    GPT2LMHeadModel,
    GPT2Tokenizer
)
import torch

class SentimentTextGenerator:
    def __init__(self):
        """Initialize sentiment analyzer and text generator models."""
        print("Loading enhanced models... This may take a moment.")
        
        # Load sentiment analysis model
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        # Load text generation model (using regular GPT-2 for better quality)
        model_name = "gpt2"  # Better quality than distilgpt2
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.generator = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Set padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.generator.config.pad_token_id = self.tokenizer.eos_token_id
        
        print("Models loaded successfully!")
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of input text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            dict: Sentiment label and confidence score
        """
        result = self.sentiment_analyzer(text)[0]
        sentiment = "positive" if result['label'] == "POSITIVE" else "negative"
        return {
            "sentiment": sentiment,
            "confidence": result['score']
        }
    
    def create_enhanced_prompt(self, user_prompt, sentiment):
        """
        Create a more effective sentiment-specific prompt with examples.
        
        Args:
            user_prompt (str): Original user prompt
            sentiment (str): Detected or selected sentiment
            
        Returns:
            str: Enhanced prompt with sentiment examples
        """
        # More detailed sentiment conditioning with examples
        sentiment_templates = {
            "positive": {
                "prefix": "Write an uplifting and optimistic story.",
                "context": "The writing should be cheerful, inspiring, and filled with hope. Use positive language and describe favorable outcomes.",
                "starter": f"Here is a positive story: {user_prompt}"
            },
            "negative": {
                "prefix": "Write a somber and melancholic story.",
                "context": "The writing should be pessimistic, disappointing, and filled with hardship. Use negative language and describe unfavorable situations.",
                "starter": f"Here is a sad story: {user_prompt}"
            },
            "neutral": {
                "prefix": "Write an objective and balanced story.",
                "context": "The writing should be factual, unbiased, and informative without emotional coloring.",
                "starter": f"Here is a factual account: {user_prompt}"
            }
        }
        
        template = sentiment_templates.get(sentiment, sentiment_templates["neutral"])
        
        # Create structured prompt
        full_prompt = f"{template['prefix']} {template['context']} {template['starter']}\n\n"
        
        return full_prompt
    
    def generate_text(self, prompt, sentiment, max_length=200, temperature=0.8, num_return_sequences=3):
        """
        Generate text with improved quality and sentiment alignment.
        
        Args:
            prompt (str): Input prompt
            sentiment (str): Sentiment to align with
            max_length (int): Maximum length of generated text
            temperature (float): Sampling temperature
            num_return_sequences (int): Generate multiple candidates and pick best
            
        Returns:
            str: Best generated text
        """
        # Create enhanced prompt
        full_prompt = self.create_enhanced_prompt(prompt, sentiment)
        
        # Encode the prompt
        inputs = self.tokenizer.encode(full_prompt, return_tensors="pt", max_length=512, truncation=True)
        
        # Create attention mask
        attention_mask = torch.ones(inputs.shape, dtype=torch.long)
        
        # Adjust parameters based on sentiment for better alignment
        if sentiment == "positive":
            temperature_adj = temperature * 0.9  # Slightly more focused
            top_p = 0.92
            repetition_penalty = 1.15
        elif sentiment == "negative":
            temperature_adj = temperature * 0.95  # More controlled
            top_p = 0.90
            repetition_penalty = 1.2
        else:  # neutral
            temperature_adj = temperature * 0.85  # Most focused
            top_p = 0.88
            repetition_penalty = 1.1
        
        # Generate multiple candidates
        with torch.no_grad():
            outputs = self.generator.generate(
                inputs,
                attention_mask=attention_mask,
                max_length=min(len(inputs[0]) + max_length, 512),
                min_length=len(inputs[0]) + 30,  # Ensure minimum generation
                temperature=temperature_adj,
                top_p=top_p,
                top_k=50,
                do_sample=True,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=3,
                early_stopping=True,
                length_penalty=1.2  # Encourage longer outputs
            )
        
        # Decode all candidates
        candidates = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            # Remove the prompt
            if full_prompt in text:
                text = text.replace(full_prompt, "").strip()
            elif len(text) > len(full_prompt):
                text = text[len(full_prompt):].strip()
            
            # Clean up
            text = self.clean_generated_text(text)
            if len(text.split()) > 15:  # Only consider substantial generations
                candidates.append(text)
        
        if not candidates:
            return "Unable to generate quality text. Please try again with different parameters."
        
        # Select best candidate based on sentiment alignment
        best_text = self.select_best_candidate(candidates, sentiment)
        
        return best_text
    
    def clean_generated_text(self, text):
        """
        Clean and format generated text.
        
        Args:
            text (str): Raw generated text
            
        Returns:
            str: Cleaned text
        """
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Ensure it ends with proper punctuation
        if text and text[-1] not in '.!?':
            # Find last sentence
            for punct in ['.', '!', '?']:
                if punct in text:
                    text = text[:text.rfind(punct)+1]
                    break
        
        return text.strip()
    
    def select_best_candidate(self, candidates, sentiment):
        """
        Select the best text from candidates based on sentiment alignment.
        
        Args:
            candidates (list): List of generated texts
            sentiment (str): Target sentiment
            
        Returns:
            str: Best matching text
        """
        best_score = -1
        best_text = candidates[0]
        
        for candidate in candidates:
            # Analyze sentiment of generated text
            try:
                gen_sentiment = self.analyze_sentiment(candidate)
                
                # Calculate alignment score
                if gen_sentiment['sentiment'] == sentiment:
                    # Prefer candidates that match sentiment with high confidence
                    score = gen_sentiment['confidence']
                else:
                    # Penalize mismatched sentiment
                    score = 1 - gen_sentiment['confidence']
                
                # Also consider length (prefer longer, more complete texts)
                length_score = min(len(candidate.split()) / 100, 1.0)
                final_score = score * 0.7 + length_score * 0.3
                
                if final_score > best_score:
                    best_score = final_score
                    best_text = candidate
            except:
                continue
        
        return best_text
    
    def generate_with_auto_sentiment(self, prompt, max_length=200, temperature=0.8):
        """
        Automatically detect sentiment and generate text.
        
        Args:
            prompt (str): Input prompt
            max_length (int): Maximum length of generated text
            temperature (float): Sampling temperature
            
        Returns:
            tuple: (generated_text, sentiment_info)
        """
        # Analyze sentiment
        sentiment_info = self.analyze_sentiment(prompt)
        sentiment = sentiment_info['sentiment']
        
        # Generate text
        generated_text = self.generate_text(prompt, sentiment, max_length, temperature)
        
        return generated_text, sentiment_info