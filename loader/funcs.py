def ensure_size(sequence, desired_size, token):
    cur_size = len(sequence)
    if cur_size <= desired_size:
        return [token] * (desired_size - cur_size) + sequence
    else:  # cur_size > cur_size, need slice
        return sequence[-desired_size:]
