"""Stopwords list and utility functions for text processing."""

# Common English stopwords
STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
    'have', 'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how',
    'their', 'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so',
    'some', 'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two',
    'more', 'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been',
    'call', 'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did',
    'get', 'come', 'made', 'may', 'part', 'i', 'me'
}


def is_stopword(token: str) -> bool:
    """
    Check if a token is a stopword.
    
    Args:
        token: The token to check
        
    Returns:
        True if the token is a stopword, False otherwise
    """
    if not token:
        return False
    
    # Convert to lowercase and strip whitespace for comparison
    clean_token = token.strip().lower()
    return clean_token in STOPWORDS