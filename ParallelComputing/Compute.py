
from timeit import default_timer as time
from multiprocessing import Process, Manager, Pool
import multiprocessing

processors = multiprocessing.cpu_count()

def asyncDictProcessing(function, processing_list, function_name = None, *args):
    
    if function_name:
        print("Starting: {}".format(function_name))
        
    start = time()
    pool = Pool(processes = multiprocessing.cpu_count())
    results = [pool.apply_async(function, args = (key, val, )) for key, val in processing_list.items()]
    output = [p.get() for p in results]
    pool.close()
    pool.join()
    end = time()
    
    if function_name:
        print("Completed: {}".format(function_name))
        print("Time required: {}".format(end-start))
    else:
        print("Completion time: {}".format(end-start))
        
    return output 

def asyncListProcessing(function, processing_list, function_name = None, *args):
    
    if function_name:
        print("Starting: {}".format(function_name))
        
    start = time()
    pool = Pool(processes = multiprocessing.cpu_count())
    results = [pool.apply_async(function, args = (val, )) for val in processing_list]
    output = [p.get() for p in results]
    pool.close()
    pool.join()
    end = time()
    
    if function_name:
        print("Completed: {}".format(function_name))
        print("Time required: {}".format(end-start))
    else:
        print("Completion time: {}".format(end-start))
        
    return output 


def asyncDFListProcessing(function, df, lst, function_name = None, *args):
    
    if function_name:
        print("Starting: {}".format(function_name))
    
    print("This could take several minutes.")
    
    start = time()
    pool = Pool(processes = multiprocessing.cpu_count())
    results = [pool.apply_async(function, args = (df, val,  )) for val in lst]
    output = [p.get() for p in results]
    pool.close()
    pool.join()
    end = time()

    if function_name:
        print("Completed: {}".format(function_name))
        print("Time required: {}".format(end-start))
    else:
        print("Completion time: {}".format(end-start))

    return output


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]