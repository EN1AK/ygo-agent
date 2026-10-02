"""Versioned lossless envelopes: never reinterpret critic V as action Q."""
import copy
import hashlib
import json


def canonical_bytes(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def upgrade_legacy_search(report):
    if report.get('schema')!='ygo-counterfactual-search-v1':
        raise ValueError('Unsupported legacy search schema')
    # Keep every number and index in the original payload. In particular do not
    # normalize visits, rename edge Q to V, or remove blocked-root-action audits.
    payload=copy.deepcopy(report)
    payload_hash=hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return {'schema':'ygo-search-audit-v1','information_mode':'oracle_exact_state',
            'fair_play_eligible':False,'source_schema':report['schema'],
            'source_payload_sha256':payload_hash,
            'methods':[key for key in ('terminal_enumeration','leaf_value_puct','candidate_rollout') if key in report],
            'source_payload':payload}


def validate_envelope(envelope):
    if envelope.get('schema')!='ygo-search-audit-v1':
        raise ValueError('Unknown audit envelope schema')
    if hashlib.sha256(canonical_bytes(envelope['source_payload'])).hexdigest()!=envelope['source_payload_sha256']:
        raise ValueError('Search audit payload changed')
    if envelope.get('information_mode')=='oracle_exact_state' and envelope.get('fair_play_eligible'):
        raise ValueError('Oracle audit cannot be fair-play evidence')
    return envelope
