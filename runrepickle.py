from deeplearning.fileinput import repickle, pkl_dir

if __name__ == '__main__':
    repickle(pkl_dir, n_proc=8)