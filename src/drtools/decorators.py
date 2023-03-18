""" 
This module was created to generate 
useful Decorators.

"""

def start_end_log(method):
    """Decorator to print name of execution method before and 
    after method execution.

    Parameters
    ----------
    method : str
        The method name that will be executed.
    """
    def decorator(f):
        def wrapper(self, *args, **kwargs):
            if self.LOGGER is not None:
                self.LOGGER.debug(f'Executing {method}()...')
                
            # execution
            response = f(self, *args, **kwargs)
            
            if self.LOGGER is not None:
                self.LOGGER.debug(f'Executing {method}()... Done!')
                
            return response
        wrapper.__doc__ = f.__doc__
        return wrapper
    return decorator