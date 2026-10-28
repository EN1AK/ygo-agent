"""Bind chain actions to physical cards/effects rather than container indices."""


def chain_choice(prompt,index):
    from ygoai.rl.human_replay import card_spec
    if prompt['op']!=16:raise ValueError('not a chain prompt')
    if index==-1:
        if prompt['forced']:raise ValueError('forced chain cannot be declined')
        return dict(act='Cancel')
    if not 0<=index<len(prompt['cards']):raise ValueError('chain index outside prompt')
    card=prompt['cards'][index];desc=card['description']
    if desc<10000:
        code=card['code'] if desc==0 else 0;effect=desc
    else:
        if desc&15>=14:raise ValueError('unsupported effect description')
        code=desc>>4;effect=10010+(desc&15)
    return dict(act='Activate',spec=card_spec(card,prompt['player']),code=code,effect=effect)


def chosen_chain_action(decision):
    action=decision['menu'][decision['action']]
    keys=('act',) if action['act']=='Cancel' else ('act','spec','code','effect')
    return {k:action[k] for k in keys}


def chain_response(prompt,choice):
    if choice['act']=='Cancel':
        chain_choice(prompt,-1)
        return (-1).to_bytes(4,'little',signed=True)
    matches=[i for i in range(len(prompt['cards'])) if chain_choice(prompt,i)==choice]
    if len(matches)!=1:raise ValueError('ambiguous or missing semantic chain choice')
    return matches[0].to_bytes(4,'little',signed=True)


def canonical_responses(record):
    decisions={d['core_response_count']:d for d in record['decisions']
        if d['message']==16 and d.get('chain_prompt') is not None and 'core_response_count' in d}
    result=[]
    for index,response in enumerate(record['core_responses']):
        if index not in decisions:
            result.append(response)
            continue
        decision=decisions[index];choice=chosen_chain_action(decision)
        raw=int.from_bytes(bytes.fromhex(response['data']),'little',signed=True)
        if chain_choice(decision['chain_prompt'],raw)!=choice:
            raise ValueError('recorded raw chain response does not select its declared action')
        result.append(dict(kind='semantic-chain-choice',player=decision['player'],choice=choice))
    return result
