import re
from typing import Dict, List, Optional, Tuple
import spacy
from pydeidentify import Deidentifier

def create_classification_prompt(comment: str) -> str:
    """Creates the classification prompt for analyzing Reddit comments about homelessness."""
    return f"""You are an expert in social behavior analysis. Your task is to analyze Reddit comments about homelessness and categorize them according to specific criteria.

DEFINITIONS:

1. Comment Types (select all that apply):
   - Ask a Genuine Question: The speaker asks a sincere question about homelessness or related issues
   - Ask a Rhetorical Question: The speaker asks a question not intended to be answered, often to make a point
   - Provide a Fact or Claim: The speaker provides a factual statement or claim about homelessness
   - Provide an Observation: The speaker shares an observation about homelessness or related situations
   - Express their Opinion: The speaker expresses their own views or feelings about homelessness
   - Express Others Opinions: The speaker describes or references the views or feelings of others about homelessness

2. Critique Categories (select all that apply):
   - Money Aid Allocation: Discussion of financial resources, aid distribution, or resource allocation for homelessness
   - Government Critique: Criticism of government policies, laws, or political approaches to homelessness
   - Societal Critique: Criticism of social norms, systems, or societal attitudes toward homelessness

3. Response Categories (select all that apply):
   - Solutions/Interventions: Discussion of specific solutions, interventions, or charitable actions

4. Perception Types (select all that apply):
   - Personal Interaction: Direct personal experiences with PEH
   - Media Portrayal: Discussion of PEH as portrayed in media
   - Not in my Backyard: Opposition to local homelessness developments
   - Harmful Generalization: Negative stereotypes about PEH
   - Deserving/Undeserving: Judgments about who deserves help

5. racist Classification:
   - Yes: Contains explicit or implicit racial bias
   - No: No racial bias present

INSTRUCTIONS:
1. Read the comment carefully
2. Analyze it according to the categories above
3. Provide your analysis in the exact format below
4. Include a brief reasoning for your classification

FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:

Comment Type: [ask a genuine question, ask a rhetorical question, provide a fact or claim, provide an observation, express their opinion, express others opinions]
Critique Category: [money aid allocation, government critique, societal critique]
Response Category: [solutions/interventions]
Perception Type: [personal interaction, media portrayal, not in my backyard, harmful generalization, deserving/undeserving]
racist: [Yes/No]
Reasoning: [brief explanation]

Comment to analyze:
\"\"\" {comment} \"\"\"

Analysis:"""

def create_mitigation_prompt(comment: str, classification: str = "") -> str:
    """Creates the mitigation prompt for rephrasing biased comments."""
    return f"""You are a social behavior expert. Rephrase the following Reddit comment to remove any bias against people experiencing homelessness. Explain your reasoning for the changes you made.

Format your response exactly as follows:

NEW COMMENT: [your rephrased comment]
REASONING: [your explanation]

Comment:
\"\"\"{comment}\"\"\"

Analysis:"""

def create_mitigation_y_n_classification_prompt(comment: str) -> str:
    """Creates a simple yes/no prompt to check if a comment is biased."""
    return f"""You are a social behavior expert. Analyze the following Reddit comment for bias against people experiencing homelessness (PEH).

Comment:
\"\"\"{comment}\"\"\"

Is this comment biased against PEH? Reply with Yes or No only.

Analysis:"""

def create_recheck_prompt(comment: str) -> str:
    """Creates a prompt to check if a mitigated comment is still biased."""
    return f"""You are a social behavior expert. Analyze the following revised Reddit comment for any remaining bias against people experiencing homelessness.

Comment:
\"\"\"{comment}\"\"\"

Is this comment still biased against PEH? Reply with Yes or No only.

Analysis:"""

def clean_text(text: str) -> str:
    """Cleans text by removing special characters and normalizing whitespace."""
    if not isinstance(text, str):
        return ""
    # Keep only alphanumeric and spaces, convert to lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text.lower())
    return ' '.join(text.split())  # Normalize whitespace

def extract_field(text: str, field_name: str) -> str:
    """Extracts a specific field from the model's response text."""
    if not isinstance(text, str):
        return ""
    
    # Find the field in the text
    pattern = rf"{field_name}:\s*(.*?)(?:\n(?!\w+:)|$)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    
    return match.group(1).strip()

def extract_mitigation_results(mitigation_output: str) -> Tuple[str, str]:
    """Extracts the new comment and reasoning from the mitigation output."""
    # Find the last NEW COMMENT: in the text
    last_new_comment = mitigation_output.rfind("NEW COMMENT:")
    if last_new_comment == -1:
        return mitigation_output, "Format error in model response"
    
    # Get everything after the last NEW COMMENT:
    text_after_new_comment = mitigation_output[last_new_comment + len("NEW COMMENT:"):]
    
    # Split on REASONING: if it exists
    if "REASONING:" in text_after_new_comment:
        new_comment, reasoning = text_after_new_comment.split("REASONING:", 1)
        return new_comment.strip(), reasoning.strip()
    else:
        return text_after_new_comment.strip(), "No reasoning provided"

def extract_classification_results(classification_output: str) -> str:
    """Extracts the classification analysis from the model's response."""
    analysis_start = classification_output.find("Analysis:")
    if analysis_start != -1:
        return classification_output[analysis_start + len("Analysis:"):].strip()
    return classification_output

def get_model_config(model_name: str) -> Dict:
    """Returns model-specific configuration parameters."""
    configs = {
        "qwen": {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "max_new_tokens": 500,
            "temperature": 0.1,
            "top_p": 0.95,
            "repetition_penalty": 1.1
        },
        "llama": {
            "model_id": "meta-llama/Llama-3.2-3B-Instruct",
            "max_new_tokens": 500,
            "temperature": 0.1,
            "top_p": 0.95,
            "repetition_penalty": 1.1
        }
    }
    return configs.get(model_name.lower(), configs["qwen"])

# Define categories
COMMENT_TYPES = [
    "ask a genuine question",
    "ask a rhetorical question",
    "provide a fact or claim",
    "provide an observation",
    "express their opinion",
    "express others opinions"
]
CRITIQUE_CATEGORIES = ["money aid allocation", "government critique", "societal critique"]
RESPONSE_CATEGORIES = ["solutions/interventions"]
PERCEPTION_TYPES = ["personal interaction", "media portrayal", "not in my backyard", "harmful generalization", "deserving/undeserving"]

def extract_flags(field_text: str, options: List[str]) -> Dict[str, int]:
    """Extracts flags from field text with strict matching."""
    flags = {opt: 0 for opt in options}

    if not field_text or not isinstance(field_text, str):
        return flags

    field_text = field_text.lower()
    if field_text.strip() in ["none", "n/a", "-", "no categories", "none applicable"]:
        return flags

    # Split on common delimiters and clean each item
    field_items = []
    for item in re.split(r'[,\n•\-]+', field_text):
        cleaned = clean_text(item)
        if cleaned:
            field_items.append(cleaned)

    # For each option, check for exact match only
    for opt in options:
        cleaned_opt = clean_text(opt)
        # Only match if the entire option is present as a complete word
        if cleaned_opt in field_items:
            flags[opt] = 1
    return flags

def create_output_row(
    comment: str,
    city: str,
    comment_text: str,
    critique_text: str,
    response_text: str,
    perception_text: str,
    racist_flag: int,
    reasoning: str,
    raw_response: str
) -> Dict:
    """Creates a standardized output row with all fields and flags."""
    output_row = {
        "Comment": comment,
        "City": city,
        "Comment Type": comment_text,
        "Critique Category": critique_text,
        "Response Category": response_text,
        "Perception Type": perception_text,
        "racist": "Yes" if racist_flag else "No",
        "Reasoning": reasoning,
        "Raw Response": raw_response
    }
    
    # Add all flag columns
    for category, flags in [
        ("Comment", extract_flags(comment_text, COMMENT_TYPES)),
        ("Critique", extract_flags(critique_text, CRITIQUE_CATEGORIES)),
        ("Response", extract_flags(response_text, RESPONSE_CATEGORIES)),
        ("Perception", extract_flags(perception_text, PERCEPTION_TYPES))
    ]:
        for flag, value in flags.items():
            output_row[f"{category}_{flag}"] = value
    
    # Add racist flag
    output_row["Racist_Flag"] = racist_flag
    
    return output_row

def load_spacy_model():
    try:
        # Load English language model
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        print("Downloading spaCy model...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

def deidentify_text(text, nlp=None):
    if not isinstance(text, str):
        return ""
    # First, use pydeidentify
    deidentifier = Deidentifier()
    text = str(deidentifier.deidentify(text))  # Convert DeidentifiedText to string
    
    # Then, apply custom regex/spaCy logic for further deidentification
    if nlp is None:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    
    # Process text with spaCy
    doc = nlp(text)
    deidentified = text
    
    # Custom patterns for domain-specific terms
    location_patterns = [
        (r'\b(?:St\.|Saint)\s+[A-Za-z]+\s+(?:County|Parish|City|Town)\b', '[LOCATION]'),
        (r'\b(?:Low|High)\s+Barrier\s+(?:Homeless|Housing)\s+Shelter\b', '[INSTITUTION]'),
        (r'\b(?:Homeless|Housing)\s+Shelter\b', '[INSTITUTION]'),
        (r'\b(?:Community|Resource)\s+Center\b', '[INSTITUTION]'),
        (r'\b(?:Public|Private)\s+(?:School|University|College)\b', '[INSTITUTION]'),
        (r'\b(?:Medical|Health)\s+Center\b', '[INSTITUTION]'),
        (r'\b(?:Police|Fire)\s+Department\b', '[INSTITUTION]'),
        (r'\b(?:City|County|State)\s+Hall\b', '[INSTITUTION]'),
        (r'\b(?:Public|Private)\s+(?:Library|Park|Garden)\b', '[INSTITUTION]'),
        (r'\b(?:Shopping|Retail)\s+Mall\b', '[INSTITUTION]'),
        (r'\b(?:Bus|Train|Subway)\s+Station\b', '[INSTITUTION]'),
        (r'\b(?:Airport|Harbor|Port)\b', '[INSTITUTION]'),
        (r'\b(?:Street|Avenue|Road|Boulevard|Drive|Lane|Place|Court|Circle|Way)\b', '[STREET]'),
        (r'\b(?:North|South|East|West|N|S|E|W)\s+(?:Street|Avenue|Road|Boulevard|Drive|Lane|Place|Court|Circle|Way)\b', '[STREET]'),
        (r'\b(?:First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth)\s+(?:Street|Avenue|Road|Boulevard|Drive|Lane|Place|Court|Circle|Way)\b', '[STREET]'),
        (r'\b(?:Main|Broad|Market|Park|Church|School|College|University|Hospital|Library)\s+(?:Street|Avenue|Road|Boulevard|Drive|Lane|Place|Court|Circle|Way)\b', '[STREET]'),
    ]
    
    # Apply location patterns first
    for pattern, replacement in location_patterns:
        deidentified = re.sub(pattern, replacement, deidentified, flags=re.IGNORECASE)
    
    # Replace named entities
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'GPE', 'LOC', 'ORG', 'DATE', 'TIME']:
            if ent.label_ == 'PERSON':
                replacement = '[PERSON]'
            elif ent.label_ in ['GPE', 'LOC']:
                replacement = '[LOCATION]'
            elif ent.label_ == 'ORG':
                replacement = '[ORGANIZATION]'
            elif ent.label_ == 'DATE':
                replacement = '[DATE]'
            elif ent.label_ == 'TIME':
                replacement = '[TIME]'
            deidentified = deidentified.replace(ent.text, replacement)
    
    # Additional patterns for emails, phones, etc.
    patterns = {
        r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}': '[PHONE]',
        r'\+\d{1,2}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}': '[PHONE]',
        r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}': '[PHONE]',
        r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}': '[PHONE]',
        r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[^\s]*)?': '[URL]',
        r'www\.(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[^\s]*)?': '[URL]',
        r'(?:[-\w.]|(?:%[\da-fA-F]{2}))+\.(?:com|org|net|edu|gov|mil|biz|info|mobi|name|aero|asia|jobs|museum)(?:/[^\s]*)?': '[URL]',
        r'\[URL\](?:/[^\s]*)?': '[URL]',
        r'\[URL\]/search\?[^\s]*': '[URL]',
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b': '[EMAIL]',
        r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b': '[IP]',
        r'\b\d{5}(?:-\d{4})?\b': '[ZIP]',
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b': '[DATE]',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b': '[DATE]',
    }
    
    # Apply additional patterns
    for pattern, replacement in patterns.items():
        deidentified = re.sub(pattern, replacement, deidentified)
    
    # Clean up any remaining URL-like or location patterns
    deidentified = re.sub(r'\[URL\]/[^\s]+', '[URL]', deidentified)
    deidentified = re.sub(r'\[URL\]\[URL\]', '[URL]', deidentified)
    deidentified = re.sub(r'\[LOCATION\]/[^\s]+', '[LOCATION]', deidentified)
    deidentified = re.sub(r'\[LOCATION\]\[LOCATION\]', '[LOCATION]', deidentified)
    deidentified = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', deidentified)
    
    return deidentified

FEW_SHOT_REDDIT_PROMPT_TEXT = '''Sentance: Are you implying that local police beat panhandlers with batons?  Because they don't .
Comment Type: [ask a genuine question, provide a fact or claim]
Critique Category: [societal critique]
Response Category: []
Perception Type: []
racist: [No]

Sentance: Most comments are saying how great it is to homeless (and it usually is) but are ignoring or unaware of the ***type*** of homeless they plan to [STREET] here.  *Drug addicts and people with mental issues.*  If it were more homes for homeless and/or low income families, I wouldn't think twice about it but I'm very concerned about a facility housing drug addicts and people with mental issues just a couple hundred feet from a school in the middle of a residential neighborhood.
Comment Type: [express their opinion, express others opinions]
Critique Category: []
Response Category: []
Perception Type: [not in my backyard, harmful generalization, deserving/undeserving]
racist: [No]

Sentance: What is up with the pots and pans?  What homeless or trafficked person needs those?  Oh wait!  She needs some.  Send her 50 sets.  She can keep one and sell the rest!  What a piece of 💩
Comment Type: [ask a rhetorical question, express their opinion]
Critique Category: [societal critique]
Response Category: []
Perception Type: [not in my backyard, harmful generalization]
racist: [No]

Sentance: "I live here too [ORGANIZATION][ORGANIZATION][ORGANIZATION]
Fuck the homeless"
Comment Type: [express their opinion]
Critique Category: []
Response Category: []
Perception Type: [not in my backyard, harmful generalization, deserving/undeserving]
racist: [No]

Sentance: "I won't support organizations that are homophobic personally. I clearly stated that others can make their own choices. I then brought up a very real issue in [ORGANIZATION]. I'm a Social Worker. I've worked directly with [ORGANIZATION] in the past. They are very religious. It is what it is. You overreacted to my post IMO. I'm not that important. Its just my opinion. But yeah, I'm not okay with discrimination, so personally I would not work for nor support [PERSON]. I know far too many GLBTQIA+ and Trans individuals that have struggled in [ORGANIZATION] because of discrimination from places like this. Trans houseless individuals in particular are often sexually assaulted around here when they start engaging in services. Its a problem. >""Get over it"" **No.**"
Comment Type: [provide a fact or claim, provide an observation, express their opinion]
Critique Category: [societal critique]
Response Category: []
Perception Type: []
racist: [No]
'''



