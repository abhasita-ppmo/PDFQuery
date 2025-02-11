from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def answer_question(question, context):
    if not context.strip():
        return "No context found"

    result = qa_pipeline(question=question, context=context)
    confidence = result['score']

    if confidence < 0.2:
        return "No context found"
    return result['answer']