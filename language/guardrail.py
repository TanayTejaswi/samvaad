import re
import logging

logger = logging.getLogger("samvaad.language.guardrail")

# A small list of allowed function words that LLMs can add to make grammar fluent.
# We do not allow nouns, verbs, or adjectives to be added if they aren't in the gloss.
ALLOWED_FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "is", "am", "are", "was", "were", 
    "be", "being", "been", "has", "have", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "up", "about", 
    "into", "over", "after", "I", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "their", "our",
    "what", "who", "where", "when", "why", "how", "this", "that", "these", "those",
    "it's", "i'm", "i've", "i'll", "don't", "can't", "please", "yes", "no", "not"
}

def clean_word(word: str) -> str:
    """Removes punctuation and lowercases."""
    return re.sub(r'[^\w\s]', '', word).strip().lower()

def check_hallucination(gloss_sequence: str, generated_text: str) -> bool:
    """
    Returns True if the generated text contains hallucinated content words.
    Returns False if the text is safe.
    """
    # 1. Normalize the glosses (split by space or hyphen)
    raw_glosses = re.split(r'[\s\-]+', gloss_sequence)
    allowed_content = {clean_word(g) for g in raw_glosses if g}
    
    # Also add standard synonyms/lemmas for glosses (Very basic stemmer)
    # E.g. "GO" allows "going", "goes", "gone"
    extended_content = set(allowed_content)
    for g in allowed_content:
        if g == "go": extended_content.update(["going", "goes", "gone"])
        if g == "eat": extended_content.update(["eating", "eats", "ate", "eaten"])
        if g == "see": extended_content.update(["seeing", "sees", "saw", "seen"])
        if g == "buy": extended_content.update(["buying", "buys", "bought"])
        if g == "meet": extended_content.update(["meeting", "meets", "met"])
        if g == "need": extended_content.update(["needs", "needed", "needing"])
        if g == "want": extended_content.update(["wants", "wanted", "wanting"])
        if g == "help": extended_content.update(["helps", "helped", "helping"])
        if g == "finish": extended_content.update(["finishes", "finished", "finishing"])
        if g == "beautiful": extended_content.update(["beauty"])
        if g == "happy": extended_content.update(["happiness"])
        if g == "pain": extended_content.update(["hurt", "hurts"])
        
    # 2. Extract words from generated text
    gen_words = [clean_word(w) for w in generated_text.split() if w]
    
    # 3. Check for violations
    for word in gen_words:
        if not word:
            continue
        # If it's a safe function word, it's fine
        if word in ALLOWED_FUNCTION_WORDS:
            continue
        # If it matches a gloss root/synonym, it's fine
        if word in extended_content:
            continue
            
        # Hallucination detected!
        logger.warning(f"Guardrail triggered! Unapproved word: '{word}' in translation: '{generated_text}'")
        return True
        
    return False

def fallback_template_join(gloss_sequence: str) -> str:
    """
    A dumb but 100% safe fallback that just joins the glosses.
    It capitalizes the first letter and adds a period.
    """
    words = gloss_sequence.replace('-', ' ').lower().split()
    if not words:
        return ""
    
    sentence = " ".join(words)
    sentence = sentence.capitalize() + "."
    return sentence
