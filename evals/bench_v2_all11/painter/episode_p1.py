from collections import namedtuple
import hashlib
import numpy as np
from tools import bench_episode_v1 as BE
Inputs = namedtuple('Inputs', ['demo_input', 'demo_target', 'query_input'])

def pixel_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def episode(pack, task, q):
    rows = [i for i, r in enumerate(pack['records']) if r['task'] == task]
    records = [pack['records'][i] for i in rows]
    sources = [str(r.get('source')) for r in records]
    same = [i for i, s in enumerate(sources) if s == sources[q]]
    if len(same) < 2:
        raise ValueError('No same-source demonstration')
    rng = np.random.default_rng([42, BE.task_seed(task), q, 1])
    cid = 0
    if BE.family_of(task) == 'palette':
        pool = BE.convention_pools(task)[1]
        cid = int(pool[int(rng.integers(0, len(pool)))])
    demo = [int(i) for i in BE.draw_demo_indices(len(rows), q, 1, rng, candidates=same)]
    # Frozen benchmark coding includes query labels in palette assignment.
    # Only the resulting demo target, demo input and query input reach the model.
    coded = BE.code_episode([pack['masks'][rows[demo[0]]], pack['masks'][rows[q]]], task, cid)
    inputs = Inputs(pack['images'][rows[demo[0]]], coded[0], pack['images'][rows[q]])
    digest = hashlib.sha256()
    for tile in (inputs.demo_input, inputs.demo_target, inputs.query_input, coded[1]):
        digest.update(np.ascontiguousarray(tile).tobytes())
    binding = dict(task=task, q=q, p=1, cid=cid, query_global=rows[q],
                   query_record=records[q]['record_id'], query_source=sources[q],
                   demo_global=[rows[demo[0]]], demo_records=[records[demo[0]]['record_id']],
                   episode_pixels_sha256=digest.hexdigest(),
                   query_input_pixels_sha256=pixel_sha(inputs.query_input),
                   query_target_pixels_sha256=pixel_sha(coded[1]))
    return inputs, binding
