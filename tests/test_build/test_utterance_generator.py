from fastworkflow.build.utterance_generator import generate_utterances

def test_generate_utterances_method_no_params():
    utterances = generate_utterances('User', 'get_details', [], is_property=False)
    # Utterance generator returns empty list - GenAI postprocessor adds utterances later
    assert utterances == []

def test_generate_utterances_method_with_params():
    params = [{'name': 'user_id'}, {'name': 'email'}]
    utterances = generate_utterances('User', 'update_email', params, is_property=False)
    # Utterance generator returns empty list - GenAI postprocessor adds utterances later
    assert utterances == []

def test_generate_utterances_property():
    utterances = generate_utterances('User', 'email', [], is_property=True)
    # Utterance generator returns empty list - GenAI postprocessor adds utterances later
    assert utterances == [] 