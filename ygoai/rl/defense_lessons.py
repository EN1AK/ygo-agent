"""Admit a declared public-response demonstration, never a global bad-action label."""
import copy


def public_top(events, start):
    top = None
    for event in events[start:]:
        if event['op'] == 74:
            top = None
        elif event['op'] == 70:
            raw = bytes.fromhex(event['raw'])
            if len(raw) != 17:
                raise ValueError('Unexpected chaining layout')
            top = (int.from_bytes(raw[1:5], 'little'), raw[9])
    return top


def admit(positive, negative, root_context, candidate, plan):
    assert positive['schema'] == negative['schema'] == 'ygo-elfnote-functional-recovery-v1'
    assert positive['probe_mode'] == 'scripted' and negative['probe_mode'] == 'pass'
    for key in ['definition', 'case', 'core_seed', 'probe_start_step', 'probe_event_start']:
        assert positive[key] == negative[key], key
    assert not positive['training_ready'] and not negative['training_ready']
    assert positive['status'] == 'success' and negative['status'] == 'goal_not_reached'
    assert positive['functional_evidence']['completed'] and positive['functional_evidence']['success']
    assert positive['functional_evidence']['challenge_negated']
    assert negative['functional_evidence']['completed'] and not negative['functional_evidence']['success']
    assert not negative['functional_evidence']['challenge_negated']
    assert positive['intervention_evidence']['success'] and negative['intervention_evidence']['success']
    name = positive['case']; declared = next(c for c in plan['cases'] if c['name'] == name)
    assert candidate['case'] == name and not candidate['training_ready']
    x, y = positive['decisions'], negative['decisions']
    first = next(i for i, (a, b) in enumerate(zip(x, y)) if a != b)
    assert first == declared['root_step'] == candidate['root_step']
    assert positive['probe_start_step'] is not None and first >= positive['probe_start_step']
    assert x[:first] == y[:first]
    a, b = x[first], y[first]
    for key in ['step', 'player', 'message', 'menu', 'observation_sha256', 'rnn_sha256', 'logits', 'value', 'core_response_count']:
        assert a[key] == b[key], key
    assert a['player'] == 0 and a['message'] == 16
    assert not a['teaching'] and not b['teaching'] and not a['searchable'] and not b['searchable']
    assert a['menu'][a['action']]['act'] == 'Activate' and a['menu'][a['action']]['code'] in plan['negators']
    assert b['menu'][b['action']]['act'] == 'Cancel'
    assert (a['action'], b['action']) == (candidate['root']['correct_action'], candidate['root']['chosen_action'])
    assert max(range(len(a['logits'])), key=lambda i:a['logits'][i]) == b['action'], 'Current policy no longer chooses the canceled response'
    assert a['observation_sha256'] == candidate['root']['observation_sha256']
    assert a['rnn_sha256'] == candidate['root']['rnn_sha256']
    assert root_context['player'] == 0 and root_context['turn_player'] == 1 and root_context['message'] == 16
    for key in ['menu', 'observation_sha256', 'rnn_sha256', 'core_response_count']:
        assert root_context[key] == a[key], key
    assert positive['events'][:len(root_context['events'])] == root_context['events']
    assert root_context['event_start'] == positive['probe_event_start']
    assert public_top(root_context['events'], root_context['event_start']) == (plan['challenge'], 1)
    lesson = copy.deepcopy(positive)
    for i, decision in enumerate(lesson['decisions']):
        decision['teaching'] = i == first
        decision['searchable'] = False
    lesson.update(schema='ygo-public-response-lesson-v1', source_schema=positive['schema'],
                  training_ready=True, lesson_protocol=plan['id'], parent_sha256=plan['parent_sha256'],
                  lesson_root=first, label_semantics=plan['label_semantics'])
    return lesson, dict(case=name, root_step=first, teaching_steps=[first], current_public_top=[plan['challenge'], 1],
                        same_prefix=True, same_observation=True, same_rnn=True, local_goal_contrast=True)
