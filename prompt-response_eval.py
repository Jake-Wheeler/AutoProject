import spacy
from textblob import TextBlob
import language_tool_python

# Load the English language model for spaCy
nlp = spacy.load('en_core_web_lg')

# Define the two possible responses
topic = input("Prompt: ")
response1 = input("Response1: ")
response2 = input("Response2: ")

# Calculate the specificity of the response to the question
def compare_keywords(question, response):
    question_doc = nlp(question)
    response_doc = nlp(response)
    # Extract keywords from question and response
    question_keywords = set([token.lemma_ for token in question_doc if not token.is_stop and not token.is_punct])
    response_keywords = set([token.lemma_ for token in response_doc if not token.is_stop and not token.is_punct])
    # Expand response keywords using semantic similarity
    for response_token in response_doc:
        for question_token in question_doc:
            if response_token.has_vector and question_token.has_vector:
                print("vector")
                similarity = response_token.similarity(question_token)
                if similarity >= 0.8:
                    response_keywords.add(response_token.lemma_)
    # Calculate score
    exact_match_count = 0
    related_match_count = 0
    for r_keyword in response_keywords:
        if r_keyword in question_keywords:
            exact_match_count += 1
        else:
            for q_keyword in question_keywords:
                if nlp(r_keyword).has_vector and nlp(q_keyword).has_vector:
                    print("vector")
                    similarity = nlp(q_keyword).similarity(nlp(r_keyword))
                    if similarity >= 0.1:
                        related_match_count += 1
    score = (exact_match_count * 2 + related_match_count) / (len(question_keywords) * 2) * 10

    return score

# Define a function to calculate sensibility of the text
def calculate_sensibility(question, response):
    doc = nlp(response)
    question_doc = nlp(question)
    sentiment_score = 0
    logical_score = 0
    # 2. Check if response is relevant to the question
    response_keywords = set([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    question_keywords = set([token.lemma_ for token in question_doc if not token.is_stop and not token.is_punct])
    keyword_overlap = response_keywords.intersection(question_keywords)
    relevance_score = len(keyword_overlap) / len(question_keywords)
    # 3. Check sentiment of response
    for sent in doc.sents:
        sentiment_score += TextBlob(str(sent)).sentiment.polarity
    sentiment_score /= len(list(doc.sents))
    # 4. Check logical consistency of response
    for sent in doc.sents:
        for token in sent:
            if token.dep_ == 'mark' and token.head.pos_ != 'VERB':
                logical_score -= 0.1
            if token.dep_ in ['aux', 'auxpass'] and token.head.pos_ not in ['VERB']:
                logical_score -= 0.1
            if token.dep_ == 'cc' and token.head.pos_ not in ['VERB']:
                logical_score -= 0.1
            if token.dep_ in ['nsubj', 'nsubjpass'] and token.head.pos_ not in ['VERB']:
                logical_score -= 0.1
            if token.dep_ == 'dobj' and token.head.pos_ not in ['VERB']:
                logical_score -= 0.1
    # Calculate overall score
    total_score = relevance_score + sentiment_score + logical_score
    sensibility = (total_score + 2) / 4 * 100
    return sensibility

# Define a function to calculate plausibility of the text
# def calculate_plausibility(text):
#     doc = nlp(text)
#     plausibility = 1 - doc.similarity(nlp("plausible"))
#     plausibility = (plausibility / 2) * 100
#     return plausibility

# Preprocess the text
doc1 = nlp(response1)
doc2 = nlp(response2)
tokens1 = [token.lemma_ for token in doc1 if not token.is_stop]
tokens2 = [token.lemma_ for token in doc2 if not token.is_stop]

# Analyze the syntax and structure of the text
pos_accuracy1 = 0
dep_accuracy1 = 0
grammar_errors1 = 0

pos_accuracy2 = 0
dep_accuracy2 = 0
grammar_errors2 = 0

for token in doc1:
    if token.pos_ != '':
        pos_accuracy1 += 1
    if token.dep_ != '':
        dep_accuracy1 += 1
    if token.is_punct or token.is_space:
        continue
    if token.is_stop or token.is_alpha:
        continue
    grammar_errors1 += 1

for token in doc2:
    if token.pos_ != '':
        pos_accuracy2 += 1
    if token.dep_ != '':
        dep_accuracy2 += 1
    if token.is_punct or token.is_space:
        continue
    if token.is_stop or token.is_alpha:
        continue
    grammar_errors2 += 1

# Calculate metrics for quality
blob1 = TextBlob(response1)
readability1 = (blob1.sentiment.polarity + 1) / 2 * 100
cohesion1 = blob1.sentiment.subjectivity * 100

blob2 = TextBlob(response2)
readability2 = (blob2.sentiment.polarity + 1) / 2 * 100
cohesion2 = blob2.sentiment.subjectivity * 100

# Provide an overall rating
grammar_rating1 = (pos_accuracy1 + dep_accuracy1 - grammar_errors1) / \
                  (pos_accuracy1+dep_accuracy1+grammar_errors1) * 100
quality_rating1 = (readability1 + cohesion1) / 2

grammar_rating2 = (pos_accuracy2 + dep_accuracy2 - grammar_errors2) / \
                  (pos_accuracy2+dep_accuracy2+grammar_errors2) * 100
quality_rating2 = (readability2 + cohesion2) / 2

# Call the functions to calculate the ratings
specificity1 = compare_keywords(topic, response1)
sensibility1 = calculate_sensibility(topic, response1)
# plausibility1 = calculate_plausibility(response1)

specificity2 = compare_keywords(topic, response2)
sensibility2 = calculate_sensibility(topic, response2)
# plausibility2 = calculate_plausibility(response2)

# Rescale the overall ratings to a 0-100 scale
overall_rating1 = (sensibility1 + specificity1 + grammar_rating1 + quality_rating1) / 4
overall_rating2 = (sensibility2 + specificity2 + grammar_rating2 + quality_rating2) / 4
if overall_rating1 > overall_rating2:
    better_response = "Response 1"
    correct_response = response1
    choice_ratio = overall_rating1 - overall_rating2
elif overall_rating2 > overall_rating1:
    better_response = "Response 2"
    correct_response = response2
    choice_ratio = overall_rating2 - overall_rating1
else:
    better_response = "ERROR"
    choice_ratio = "ERROR"
    correct_response = "ERROR"

# Correct the better choice and output improvements
gram_tool = language_tool_python.LanguageTool('en-US')  # set language
gram_text = correct_response
matches = gram_tool.check(gram_text)

for match in matches:
    print(match.ruleId, match.replacements)

corrected_text = gram_tool.correct(gram_text)


# Print the results
print("THE BEST CHOICE IS: ", better_response)
print("THE RESPONSE IS {}% BETTER.".format(choice_ratio))
print("TEXT CORRECTION: ", corrected_text)

print("\nResponse 1: " + response1)
print("Grammar rating: {:.2f}%".format(grammar_rating1))
print("Quality rating: {:.2f}%".format(quality_rating1))
print("Specificity: {:.2f}%".format(specificity1))
print("Sensibility: {:.2f}%".format(sensibility1))
# print("Plausibility: {:.2f}%".format(plausibility1))


print("\nResponse 2: " + response2)
print("Grammar rating: {:.2f}%".format(grammar_rating2))
print("Quality rating: {:.2f}%".format(quality_rating2))
print("Specificity: {:.2f}%".format(specificity2))
print("Sensibility: {:.2f}%".format(sensibility2))
# print("Plausibility: {:.2f}%".format(plausibility2))

input("Press Enter to close the program...")
