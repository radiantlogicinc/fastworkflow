from fastworkflow.build.utterance_generator import generate_utterances

def test_generate_utterances_method_no_params():
    utterances = generate_utterances('User', 'get_details', [], is_property=False)
    assert any('get details user' in u for u in utterances)
    assert any('Call get_details on user' in u for u in utterances)

def test_generate_utterances_method_with_params():
    params = [{'name': 'user_id'}, {'name': 'email'}]
    utterances = generate_utterances('User', 'update_email', params, is_property=False)
    assert any('{user_id}' in u and '{email}' in u for u in utterances)
    assert any('update email user' in u for u in utterances)

def test_generate_utterances_property():
    utterances = generate_utterances('User', 'email', [], is_property=True)
    utterances_lc = [u.lower() for u in utterances]
    assert any('get email of user' in u for u in utterances_lc)
    assert any('retrieve user email' in u for u in utterances_lc)
    assert any('show user email' in u for u in utterances_lc) 