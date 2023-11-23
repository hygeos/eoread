from inspect import isclass, signature, _empty

"""
Helpers to provide garanties akin to statically compiled languages
"""

class WrongUsage(Exception):
    pass

class ClassIsFrozen(Exception):
    pass

class ClassIsAbstract(Exception):
    pass

class MethodIsAbstract(Exception):
    pass

class InterfaceException(Exception):
    pass

def freeze(my_class):
    """
    Declare a class as frozen after construction,
    Raise an error if attempting to add attributes post constructor
    """
    
    if not isclass(my_class):
        if callable(my_class):
            raise WrongUsage(f'\n\tCan only apply @freeze to a class, and \'{my_class.__name__}\' is not a class')

    def _setattr_decorator(self, key, value):
        
        if not hasattr(self, key) and hasattr(self, '__frozen'):
            raise ClassIsFrozen( f"\n\tCustom class '{type(self).__name__}' is a frozen class, cannot reassign new attributes")
        else:            
            object.__setattr__(self, key, value)
    
    my_class.__setattr__ = _setattr_decorator
    
    def _init_decorator(init_func):
        def wrapper(self, *args, **kwargs):
            init_func(self, *args, **kwargs)
            self.__frozen = True # freeze object after its constructor
        return wrapper
    
    my_class.__init__ = _init_decorator(my_class.__init__)
    
    return my_class



def abstract(myclass_or_method):
    """
    Declare a Class or method as abstract
    Raise an error if trying to call __init__() or the method
      - Abstract Classes are meant to be derived in subclasses
      - Abstract Methods are meant to be overriden in subclasses
    """
    
    def _init_abstract_decorator(init_method):
        def inner(self, *args, **kwargs):
            
            if myclass_or_method is type(self): # didn't use issubclass to allow subclass usage of superclass
                raise ClassIsAbstract( f"\n\tClass '{myclass_or_method.__name__}' is abstract and is not meant be instanced")
                
            init_method(self, *args, **kwargs)
        return inner
    
    def _abstract_decorator(my_method):
        def inner(self, *args, **kwargs):
            raise MethodIsAbstract( f"\n\tMethod '{my_method.__name__}' from class '{type(self).__name__}' is abstract and is meant to be overriden")
        return inner
    
    if isclass(myclass_or_method):
        myclass_or_method.__init__ = _init_abstract_decorator(myclass_or_method.__init__)
        return myclass_or_method
    # else: # is a method
    return _abstract_decorator(myclass_or_method)


def virtual(function):
    """
    Declare a function or method as virtual (default python behavior)
    Only has a documenting purpose
    """
    return function


def interface(function):
    """
    Declare a function or method as an Interface
    Raise an error if types passed do not match definition
    """
    if isclass(function):
        raise WrongUsage(f'\n\tCannot declare class \'{function.__name__}\' as an interface, only functions or methods can be')
        
    def wrapper(*args, **kwargs):
        
        # construct datastructures used
        expected_signature = [(i.name, i.annotation) for i in signature(function).parameters.values()]
        unnamed_params = [type(i) for i in args] 
        named_params  = [(i, type(kwargs[i])) for i in kwargs] # named parameters can only be lasts 
        default_params = function.__defaults__ or []

        # checknumber of parameters
        exp_nargs = len(expected_signature)
        act_nargs = len(unnamed_params) + len(named_params) + len(default_params)
        
        if exp_nargs > act_nargs:
            raise InterfaceException(f'\n\tFunction \'{function.__name__}\': Exepected {exp_nargs} arguments, got {act_nargs}')
        
        errors = []
        # check unnamed parameters
        for param_type in unnamed_params:
            expected_name, expected_type = expected_signature.pop(0)
            if expected_type == _empty: continue
            
            if hasattr(expected_type, '__origin__'): # workaround for defs like list[str] → list (only check base type)
                expected_type = expected_type.__origin__
                
            if not issubclass(param_type, expected_type):
                print("ERROR:", expected_name, expected_type, param_type)
                errors.append((expected_name, expected_type, param_type))
        
        expected_signature = {i[0]: i[1] for i in expected_signature}
        # check named parameters
        for param_name, param_type in named_params:
            expected_type = expected_signature[param_name]
            if expected_type == _empty: 
                continue
            
            if hasattr(expected_type, '__origin__'): # workaround for defs like list[str] → list (only check base type)
                expected_type = expected_type.__origin__
                
            if not issubclass(param_type, expected_type):
                print("ERROR:", expected_name, expected_type, param_type)
                errors.append((param_name, expected_type, param_type))
    
        # raise error if at least one mismatch
        if len(errors) != 0: # error on at least one parameter
            mess = f'\n\tFunction \'{function.__name__}\': Wrong parameters types passed:'  
            for p in errors:
                mess += (f'\n\t\tParameter \'{p[0]}\' expected: {p[1]} got {p[2]}')
            raise InterfaceException(mess)

        return function(*args, **kwargs)
    return wrapper


# class decorator
# def _constructor_decorator(myclass):
#     def _decorator(method):
        
#         def inner(self, *args, **kwargs):
#             print(f"Calling constructor of class {type(self).__name__}")
#             method(self, *args, **kwargs)
#         return inner

#     myclass.__init__ = _decorator(myclass.__init__)
    
#     return myclass