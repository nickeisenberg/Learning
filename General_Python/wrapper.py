def decor(fun):
    def wrapper(*args, **kwargs):
        print('args :', args)
        print('kwargs :', kwargs)
        return fun(*args, **kwargs)
    return wrapper

@decor
def func(x):
    print(x)
    return None

func('inside of func')
