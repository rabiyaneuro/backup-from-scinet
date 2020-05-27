def etime(comstr,varstr=""):
    import builtins
    import inspect
    frame = inspect.currentframe()
    try:
        uplocals = frame.f_back.f_locals
        builtins.__dict__.update(uplocals)
    finally:
        del frame
    from timeit import Timer
    from pickle import dumps
    reconvar="from pickle import loads;"
    for v in varstr.split(','):
        v = v.strip()
        if v:
            if v != '':
                reconvar += " "+v+"=loads("+repr(dumps(uplocals[v]))+");"
    print("Elapsed:",Timer(comstr,reconvar).timeit(20)/20,"seconds")
