import re

def compute_metrics(question, answer, context_docs):

    context_text = " ".join([doc.page_content for doc in context_docs])

    question_words = set(re.findall(r'\w+', question.lower()))
    answer_words = set(re.findall(r'\w+', answer.lower()))
    context_words = set(re.findall(r'\w+', context_text.lower()))

    # Faithfulness: how much answer overlaps with context
    faithfulness = len(answer_words & context_words) / max(len(answer_words), 1)

    # Relevance: overlap between question and answer
    answer_relevance = len(question_words & answer_words) / max(len(question_words), 1)

    # Context precision: useful info retrieved
    context_precision = len(answer_words & context_words) / max(len(context_words), 1)

    # Context recall: how much question info is in context
    context_recall = len(question_words & context_words) / max(len(question_words), 1)

    return {
        "faithfulness": round(faithfulness, 2),
        "answer_relevance": round(answer_relevance, 2),
        "context_precision": round(context_precision, 2),
        "context_recall": round(context_recall, 2)
    }