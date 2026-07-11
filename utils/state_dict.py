def extract_state_dict(checkpoint_obj):
    if not isinstance(checkpoint_obj, dict):
        raise TypeError(f'Checkpoint must be a dict, got: {type(checkpoint_obj).__name__}')
    if 'state_dict' not in checkpoint_obj:
        raise KeyError('Checkpoint missing state_dict')
    return checkpoint_obj['state_dict']
