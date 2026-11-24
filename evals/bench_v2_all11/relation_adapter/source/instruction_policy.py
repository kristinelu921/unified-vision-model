"""One immutable instruction identity shared by inference and score provenance."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
GENERIC='Apply the demonstrated image transformation.'

def prompt_map_for(condition):
    if condition=='generic_text':return {}
    if condition!='task_text_alpha1':raise ValueError('Unknown instruction condition')
    config=json.loads((ROOT/'configs/task_instructions.json').read_text())
    if config['condition']!=condition or config['alpha']!=1.0 or len(config['prompts'])!=13:
        raise ValueError('Frozen task instruction configuration differs')
    return config['prompts']

def policy(condition):
    return {'condition':condition,'task_prompts':prompt_map_for(condition),
            'generic_prompt':GENERIC,'max_sequence_length':512,'alpha':1.0}

def select_embeddings(runtime,condition,task):
    prompts=prompt_map_for(condition)
    if prompts:
        if task not in prompts:raise ValueError('Unknown public task label')
        return runtime['task_embeddings'][task]
    return runtime['prompt_embeds'],runtime['pooled_prompt_embeds']

def verify_receipt_policy(receipt):
    expected=policy(receipt['condition'])
    if (receipt.get('instruction_policy')!=expected or receipt.get('task_prompts')!=expected['task_prompts']
        or receipt.get('execution_contract',{}).get('config',{}).get('instruction_policy')!=expected):
        raise ValueError('Instruction condition differs between inference and scoring')
    return expected

def policy_from_plan(plan):
    execution=plan['autoresearch']
    expected=policy(execution['condition'])
    if execution.get('instruction_policy')!=expected or execution.get('task_prompts')!=expected['task_prompts']:
        raise ValueError('Scoring plan instruction policy differs')
    return expected
