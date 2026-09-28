"""Local offline text summarization and notes generation.

Implements rule-based extraction to provide offline notes
without requiring external cloud APIs.
"""



class NotesGenerator:
    """Offline rule-based notes generator."""

    def __init__(self) -> None:
        # Stopwords for simple extractive summarization
        self.stop_words = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for",
            "from", "has", "he", "in", "is", "it", "its", "of", "on",
            "that", "the", "to", "was", "were", "will", "with"
        }

    def extract_keywords(self, text: str, top_k: int = 5) -> list[str]:
        """Extracts top keywords from text based on frequency."""
        words = ''.join(c if c.isalnum() else ' ' for c in text.lower()).split()
        words = [w for w in words if w not in self.stop_words and len(w) > 3]
        
        freqs: dict[str, int] = {}
        for w in words:
            freqs[w] = freqs.get(w, 0) + 1
            
        sorted_words = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
        return [w[0] for w in sorted_words[:top_k]]
        
    def generate_summary(self, transcripts: list[str]) -> str:
        """Generates a brief summary of a list of transcripts."""
        if not transcripts:
            return "No conversation recorded."
            
        full_text = " ".join(transcripts)
        keywords = self.extract_keywords(full_text)
        
        if not keywords:
            return "Conversation too short to summarize."
            
        return f"Key topics discussed: {', '.join(keywords)}."
