from query_data import query_rag
from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate


EVAL_PROMPT = """
Expected Response: {expected_response}
ActualResponse: {actual_response}
---
(Answer with true or false) Does the expected response match the actual response?
"""


def query_and_validate(question: str, expected_response: str):
    actual_response = query_rag(question)
    prompt_template = ChatPromptTemplate.from_template(EVAL_PROMPT)
    prompt = prompt_template.format(expected_response=expected_response, actual_response=actual_response)

    model = Ollama(model="mistral-nemo:latest")
    evaluation_response = model.invoke(prompt)
    evaluation_response_clean = evaluation_response.lower()

    if "true" in evaluation_response_clean:
        print(f"Response: {evaluation_response_clean}")
        return True
    elif "false" in evaluation_response_clean:
        print(f"Response: {evaluation_response_clean}")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )
    
def test_monopoly():
    assert query_and_validate(
        question="How much total money does a player start with in Monopoly? (Answer with the number only)",
        expected_response="$1500"
    )

def test_ticket_to_ride_rules():
    assert query_and_validate(
        question="How many points does the longest continuous train get in Ticket to Ride? (Answer with the number only)",
        expected_response="10 points"
    )

