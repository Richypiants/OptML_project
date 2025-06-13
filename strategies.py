import torch

# The frequency stays the same throughout all the iterations of the training process
def constant_frequency(state):
    """
    Function signature will be described only here, as it is the same throughout all the frequency schedules.

    Parameters:
        state: dictionary containing the information used by the SOAP algorithm.
          When calling a schedule, SOAP passes the state dictionary as an argument.

    Returns:
      True when the frequency must be updated in the current iteration based on the received state, False otherwise.
    """

    global counter
    if "last_update" not in state:
        state["last_update"] = 0
        counter = 0

    if state['step'] % state['precondition_frequency'] == 0:
        counter += 1
        return True
    else:
        return False

# The frequency gets halved down at each update of the preconditioner, making the updates more and more sparse as the training progresses
def halving_frequency(state):
    global counter
    if "last_update" not in state:
        state["last_update"] = 0
        counter = 1
        return True

    if state['step'] - state['last_update'] >= state['precondition_frequency']:
        state['last_update'] = state['step']
        state['precondition_frequency'] = max(1, state['precondition_frequency']//2) # Clip the frequency to min 1
        counter += 1
        return True
    else:
        return False

# The frequency gets doubled at each update of the preconditioner, making the updates more and more frequent as the training progresses
def doubling_frequency(state):
    global counter
    if "last_update" not in state:
        state["last_update"] = 0
        counter = 1
        return True

    if state['step'] - state['last_update'] >= state['precondition_frequency']:
        state['last_update'] = state['step']
        state['precondition_frequency'] = state['precondition_frequency']*2  # Clip the frequency to max 1024
        counter += 1
        return True
    else:
        return False

# Same as the previous, but with a tighter clip as the maximum frequency
def doubling_frequency_clipped(state):
    global counter
    if "last_update" not in state:
        state["last_update"] = 0
        counter = 1
        return True

    if state['step'] - state['last_update'] >= state['precondition_frequency']:
        state['last_update'] = state['step']
        state['precondition_frequency'] = min(256, state['precondition_frequency']*2)  # Clip the frequency to max 1024
        counter += 1
        return True
    else:
        return False

# The frequency gets doubled at each update of the preconditioner up to a certain threshold,
# and then gets halved down to 1 once again, leading to a lot of updates at both start and end of training but not in the middle
def doubling_then_halving_frequency(state):
    global counter
    if "last_update" not in state:
        state['last_update'] = 0
        state['last_freq'] = 1
        state['halving'] = False
        counter = 1
        return True

    if state['last_freq'] >= state['precondition_frequency']:
        state['halving'] = True

    if state['step'] - state['last_update'] >= state['last_freq']:
        state['last_update'] = state['step']
        if state['halving']:
            state['last_freq'] = max(1, state['last_freq']//2)
            counter += 1
        else:
            state['last_freq'] = min(1024, state['last_freq']*2)
            counter += 1
        return True
    else:
        return False

# A fixed interval of 256 iterations is defined, and the frequency gets doubled at the end of each interval
def fixed_interval_doubling_frequency(state):
    global counter
    if "last_update" not in state:
        state['last_update'] = 0
        counter = 1
        return True

    if state['step'] - state['last_update'] >= 256:
        state['last_update'] = state['step']
        state['precondition_frequency'] = min(256, state['precondition_frequency']*2)

    if (state['step'] - state['last_update']) % state['precondition_frequency'] == 0:
        counter += 1
        return True
    else:
        return False

# There is not a real "frequency", but the preconditioner gets updated when the loss changes more than a certain threshold
def loss_dependent_frequency(state):
    global counter
    if "last_loss" not in state:
        state['last_loss'] = state['loss']
        counter = 1
        return True

    if abs(state['last_loss'] - state['loss']) >= state['precondition_frequency']:
        state['last_loss'] = state['loss']
        counter += 1
        return True
    else:
        return False