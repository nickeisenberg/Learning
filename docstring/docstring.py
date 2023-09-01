def google_doc(a: int, b: int):
    """
    This is an example of Google style.
    Args:
        param1: These need to be labeled as param1 etc. Im not sure if this
        is specific to my computer, but noice.nvim displays the docstring
        weird otherwise
        param2: This is a second param.
    
    Returns:
        This is a description of what is returned.
    
    Raises:
        KeyError: Raises an exception.
    """
    return a + b

def numpy_doc(a=1, b=2, c=3):
    """
    My numpydoc description of a kind
    of very exhautive numpydoc format docstring.
    
    Parameters
    --------------------------------------------------
    a : array_like
        the 1st param name `first`. It seem like noice.nvim displays the
        docstring nicely even if I change the name from 'a' to first etc.
    second :
        the 2nd param
    c : {'value', 'other'}, optional
        the 3rd param, by default 'value'
    
    Returns
    --------------------------------------------------
    string
        a value in a string
    
    Raises
    --------------------------------------------------
    KeyError
        when a key error
    OtherError
        when an other error
    """
    return f'{a}, {b}, {c}'

