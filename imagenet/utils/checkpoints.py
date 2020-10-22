def remove_module(checkpoint):
    """
    remove DataParallel model.module wrapper.
    """
    new_state_dict = {}
    for key in checkpoint['state_dict']:
        if str.startswith(key, "module."):
            new_key = key[7:]  # remove "module."
            new_state_dict[new_key] = checkpoint['state_dict'][key]
        else:
            new_state_dict[key] = checkpoint['state_dict'][key]

    checkpoint['state_dict'] = new_state_dict
