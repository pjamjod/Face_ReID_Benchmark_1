import time
import functools

def timeit(func):
    """
    A decorator that measures the execution time of a function
    and prints the result in milliseconds.
    """
    # @functools.wraps preserves the original function's metadata (name, docstring)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # time.perf_counter() is a high-precision clock for measuring short intervals
        start_time = time.perf_counter()
        
        # Call the original function and store its result
        result = func(*args, **kwargs)
        
        # Calculate the elapsed time
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        
        # Print the timing information
        print(f"[TIMER] '{func.__name__}' executed in {duration_ms:.2f} ms")
        
        # Return the original function's result
        return result
        
    return wrapper