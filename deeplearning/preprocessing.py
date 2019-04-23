# proprocessing for dictionary embedding model
# traverse the whole dataset and count the unique words

# r"[^\W\d_]+|\d+|[^a-zA-Z\d\s:]+"

import re
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor, as_completed
import operator
import pickle
import os

corpus_dir=Path("D:")/"Git"/"trdata"/"languages"
lan_dic = {"C": 0,
           "C#": 1,
           "C++": 2,
           "Go": 3,
           "Java": 4,
           "Javascript": 5,
           "Lua": 6,
           "Objective-C": 7,
           "Python": 8,
           "Ruby": 9,
           "Rust": 10,
           "Shell": 11}



def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=3):
    """
        A parallel version of the map function with a progress bar.

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of
                keyword arguments to function
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job.
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    # We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    # If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    # Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        # Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        # Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    # Get the results from the futures.
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out

# this function will run for an hour


def sub_list_matcher(file_list):
    # this function is given to a subprocess
    word_count={}
    for file in file_list:
        with file.open('r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                matches = re.findall(r"[^\W\d_]+|\d+|[^a-zA-Z\d\s:]+", line)
                for match in matches:
                    if match in word_count:
                        word_count[match] += 1
                    else:
                        word_count[match] = 1
    return word_count

def combine_dictionary(base, result_dicts, thres=10):
    """

    :param base:
    :param result_dicts:
    :param thres: the top-thres percentage will be retained
    :return:
    """
    for dic in result_dicts:
        for word, count in dic.items():
            try:
                base[word]+=count
            except KeyError:
                base[word]=count

    sorted_counts=sorted(base.items(), key=operator.itemgetter(1), reverse=True)
    cutoff=sorted_counts[0:len(sorted_counts)//10]
    return {k:v for k,v in cutoff}

def generate_dictionary(n_proc=8,bs=1000, save=True):
    """
    This function counts all files.
    The dictionary will be several gigabytes big with naive run and takes 1 hour, so an algorithm is necessary.
    Every time the threads converge, the dictionary is updated and keys with more than 50 times counts are retained

    :param n_proc: number of processes are spawn to count the files in parallel. Regex takes time, so CPU is the bottleneck.
    :param converge: number of files for each thread to count before converging and updating the dictionary
    :return:
    """


    word_count={}
    for dir in corpus_dir.iterdir():
        if dir.is_dir():
            print("counting", dir)
            file_list=list(dir.iterdir())
            batches_n=len(file_list)//bs
            batches=[file_list[0+bs*b:bs+bs*b] for b in range(batches_n+1)]

            results= Parallel(n_jobs=n_proc)(delayed(sub_list_matcher)(file_list) for file_list in batches)
            word_count=combine_dictionary(word_count, results)


    if save:
        with open("dic.pkl","wb+") as f:
            pickle.dump(word_count,f)
    return word_count


def select_vocabulary(size=50000):
    """
    Return the top size vocabulary, sorted by frequency, collected over training set.
    :param size:
    :return:
    """
    deeplearning_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    with open(deeplearning_dir/"dic.pkl", "rb") as f:
        hello = pickle.load(f)
    sorted_hello = sorted(hello.items(), key=operator.itemgetter(1),reverse=True)
    vocabulary=[word for word,count in sorted_hello[:size]]

    # print("done")

    return vocabulary

if __name__ == '__main__':
    ret=select_vocabulary()