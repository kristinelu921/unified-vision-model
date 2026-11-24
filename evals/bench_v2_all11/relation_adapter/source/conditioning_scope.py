"""One-episode cache lifetime. No process-global conditioning or predictions."""
from contextlib import contextmanager

import torch


@contextmanager
def prepared_relation(processors, image_emb, identity, *, heads, head_dim):
    """Bind the exact tensor passed to the pipeline; clear even after failure.

    P=1, batch=1 is intentional: the official two-image concatenation is not a
    general batched layout. Changing batching needs a separate parity check.
    """
    if image_emb.ndim != 3 or tuple(image_emb.shape[:2]) != (2, 128):
        raise ValueError('only official two-image, 128-token, batch=1 layout is qualified')
    with torch.no_grad():
        relation = image_emb.detach().clone()
    bound = []
    try:
        for processor in processors:
            if not hasattr(processor, 'bind_relation_cache'):
                raise TypeError('all adapter processors must support explicit cache lifetime')
            bound.append(processor)
            processor.bind_relation_cache(relation, identity=identity.digest(),
                                          batch_size=1, heads=heads, head_dim=head_dim)
        yield relation
    finally:
        for processor in bound:
            processor.clear_relation_cache()
