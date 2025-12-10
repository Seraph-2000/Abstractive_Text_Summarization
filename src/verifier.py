import torch
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class HallucinationVerifier:
    def __init__(self, device="cuda"):
        print("Loading NLI Verifier (roberta-large-mnli)...")
        
        # Ensure NLTK tokenizer data is available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK 'punkt' data...")
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)

        self.device = device
        # We use roberta-large-mnli because it is excellent at detecting contradiction/entailment
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
        self.model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").to(self.device)
        
        # In roberta-large-mnli: 0=Contradiction, 1=Neutral, 2=Entailment
        self.entailment_id = 2 

    def verify(self, source_text, generated_summary):
        """
        Splits summary into claims and checks each against the source text.
        Returns a list of tuples: [('FAITHFUL', "claim..."), ('HALLUCINATION', "claim...")]
        """
        # 1. Decompose summary and source into individual sentences
        claims = nltk.sent_tokenize(generated_summary)
        source_sentences = nltk.sent_tokenize(source_text)
        
        results = []
        
        if not claims:
            return [("EMPTY", "No summary generated.")]

        # 2. Verify each claim (sentence) from the summary
        for claim in claims:
            is_hallucination = True  # Assume guilty until proven innocent
            
            # 3. Check this claim against EVERY sentence in the source
            # If ANY source sentence supports it (Entailment), it is faithful.
            for source_sent in source_sentences:
                inputs = self.tokenizer(
                    source_sent,   # Premise
                    claim,         # Hypothesis
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512
                ).to(self.device)
                
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                
                predicted_class_id = logits.argmax().item()
                
                # 4. If we find support (Entailment), stop checking this claim
                if predicted_class_id == self.entailment_id:
                    is_hallucination = False
                    break 
            
            # 5. Record the verdict
            status = "HALLUCINATION" if is_hallucination else "FAITHFUL"
            results.append((status, claim))
            
        return results