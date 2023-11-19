import spacy

# Download spaCy model (English)
# python -m spacy download en

def tokenize_text(text):
    nlp = spacy.load('en')
    doc = nlp(text)
    tokens = [token.text for token in doc]
    return tokens

def pos_tagging(text):
    nlp = spacy.load('en')
    doc = nlp(text)
    pos_tags = [(token.text, token.pos_) for token in doc]
    return pos_tags

def named_entity_recognition(text):
    nlp = spacy.load('en')
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def sentiment_analysis(text):
    nlp = spacy.load('en')
    doc = nlp(text)
    sentiment_score = doc.sentiment
    return sentiment_score

def main():
    sample_text = "Natural Language Processing is fascinating. It involves the analysis of text to extract meaningful insights."

    # Tokenization
    tokens = tokenize_text(sample_text)
    print("Tokenized Words:", tokens)

    # Part-of-Speech Tagging
    pos_tags = pos_tagging(sample_text)
    print("Part-of-Speech Tags:", pos_tags)

    # Named Entity Recognition
    entities = named_entity_recognition(sample_text)
    print("Named Entities:", entities)

    # Sentiment Analysis
    sentiment_score = sentiment_analysis(sample_text)
    print("Sentiment Analysis Score:", sentiment_score)

if __name__ == "__main__":
    main()
