import numpy as np
import os
import random
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch
import sys
import pickle
from pathlib import Path
import string
from preprocessing import select_vocabulary
import re
from tqdm import tqdm
from preprocessing import parallel_process
from joblib import Parallel, delayed

pkl_dir=Path("C:\\") / "Git" / "trdata" / "pkl"
corpus_dir=Path("D:")/"Git"/"trdata"/"languages"

string_map={}
for idx,char in enumerate(string.printable):
    string_map[char]=idx


def get_array(filename="somecode.txt"):

    with open(filename,'r') as file:
        while True:
            c=file.read(1)
            if not c:
                break
            dim=ord(c)
            thisarr=np.zeros([256])
            thisarr[dim]=1
            yield thisarr


# this function is critical for generating all datasets in all plans.
# it communicates with the scraped files.
def get_data_set(lan_dic, proportion=0.2, load=True, save=False, shuffle=True):
    pkl_dir=corpus_dir.parent / "train_valid.pkl"
    if load:
        if pkl_dir.exists():
            with pkl_dir.open("rb") as f:
                train, valid = pickle.load(f)
                return train, valid


    # split data set to training and validation
    random.seed(1234)
    np.random.seed(1234)

    # train and valid are both lists of (language_script_path, file_length) tuples
    train=[]
    valid=[]

    count=1e10
    for language in lan_dic:
        cc = len(os.listdir(corpus_dir / language))
        count=min(count, cc)

    for language in lan_dic:
        tt,vv=train_valid_files_split(language,count,proportion)
        train+=tt
        valid+=vv

    # sample minibatch of files

    # get input array and target
    # the input will be a fixed length cut from the file, let's say it's 512 characters, padded if shorter
    # the output will be a one hot array denoting the target
    if shuffle:
        random.shuffle(train)
        random.shuffle(valid)

    if save:
        with pkl_dir.open("wb") as f:
            pickle.dump((train,valid),f)
    return train, valid


def train_valid_files_split(language, count, no_file_length=True, proportion=0.2):
    """
    Given a language folder, splits to train valid, calculates the file length

    :param language: name of the language
    :param proportion: proportion of validation set
    :return:
    """
    all_files=os.listdir(corpus_dir/language)
    all_files=random.sample(all_files,k=count)

    valid_size=int(len(all_files)*proportion)+1
    valid_list=random.sample(all_files,valid_size)
    train_list=[]
    for file in all_files:
        if file not in valid_list:
            language_script_path=language+'/'+file
            if no_file_length:
                flen=0
            else:
                flen=get_file_length(language_script_path)
            train_list.append((language_script_path,flen))
    for i in range(len(valid_list)):
        lsp=language+'/'+valid_list[i]
        if no_file_length:
            flen = 0
        else:
            flen = get_file_length(lsp)
        valid_list[i]=(lsp,flen)

    new_train_list=[]
    new_valid_list=[]

    if no_file_length:
        return train_list, valid_list
    else:
        for i in range(len(train_list)):
            lsp,fl=train_list[i]
            if fl>30:
                new_train_list.append(train_list[i])

        for i in range(len(valid_list)):
            lsp,fl=valid_list[i]
            if fl>30:
                new_valid_list.append(valid_list[i])

        return new_train_list, new_valid_list

def get_file_length(language_script_path):
    """

    :param language_script_path: path to a file in the language/script format.
    :return:
    """
    try:
        with open(corpus_dir/language_script_path, 'r', encoding="utf8", errors='ignore') as file:
            file_length=0
            while True:
                c = file.read(1)
                if not c:
                    break
                file_length+=1
    except UnicodeDecodeError:
        print(language_script_path)
        raise
    return file_length


def training_file_to_input_target_arrays(train_valid_point, lan_dic, input_len):
    '''

    :param train_valid_point: (language_script_path, file_len)
    :param lan_dic: dictionary of languages and their target index
    :param input_len: the size of the cut of the file to be fed in
    :return: numpy arrays
    '''
    target=np.zeros(1,dtype=np.long)
    inputs=np.zeros((input_len,100))

    scriptpath, file_len=train_valid_point
    lang,file=scriptpath.split('/')
    target_index=lan_dic[lang]
    target[0]=target_index

    # ensure that at least 20 char is read
    # no longer seek randomly
    offset=0
    with open(corpus_dir/scriptpath,'r', encoding="utf8", errors='ignore') as file:
        file.seek(offset,0)
        input_char=file.read(input_len)

    # this effectively allows you to skip any non ascii characters
    empty_count=0
    for idx,char in enumerate(input_char):
        try:
            char=char.encode('ascii',errors='ignore').decode('ascii')
            if char != '':
                inputs[idx-empty_count,string_map[char]]=1
            else:
                empty_count+=1
        except IndexError:
            print("What?")

    return inputs, target


def training_file_to_bigram(train_valid_point, lan_dic, input_len):
    '''

    :param train_valid_point: (language_script_path, file_len)
    :param lan_dic: dictionary of languages and their target index
    :param input_len: the size of the cut of the file to be fed in
    :return: numpy arrays
    '''

    target=np.zeros(1,dtype=np.long)
    inputs=np.zeros((input_len,10000))

    scriptpath, file_len=train_valid_point
    lang,file=scriptpath.split('/')
    target_index=lan_dic[lang]
    target[0]=target_index

    # ensure that at least 20 char is read
    # no longer seek randomly
    offset=0
    with open(corpus_dir/scriptpath,'r', encoding="utf8", errors='ignore') as file:
        file.seek(offset,0)
        input_char=file.read(input_len)

    # this effectively allows you to skip any non ascii characters
    for idx, char in enumerate(input_char):
        char=char.encode('ascii',errors='ignore').decode('ascii')
        if char not in string_map:
            char = ' '
        if idx==0:
            pass
        else:
            try:
                bigram_idx=string_map[lastchar]*100+string_map[char]
                inputs[idx, bigram_idx] = 1
            except KeyError:
                char=" "
                bigram_idx=string_map[lastchar]*100+string_map[char]
                inputs[idx, bigram_idx] = 1
        lastchar=char
    bigram_idx = string_map[lastchar] * 100 + string_map[" "]
    inputs[idx,bigram_idx]=1

    #
    # for idx,char in enumerate(input_char):
    #     try:
    #         char=char.encode('ascii',errors='ignore').decode('ascii')
    #         if char != '':
    #             inputs[idx-empty_count,string_map[char]]=1
    #         else:
    #             empty_count+=1
    #     except IndexError:
    #         print("What?")

    return inputs, target


def training_file_to_bigram_bow(train_valid_point, lan_dic, input_len):
    '''

        :param train_valid_point: (language_script_path, file_len)
        :param lan_dic: dictionary of languages and their target index
        :param input_len: the size of the cut of the file to be fed in
        :return: numpy arrays
        '''

    target = np.zeros(1, dtype=np.long)
    inputs = np.zeros((input_len, 10000))

    scriptpath, file_len = train_valid_point
    lang, file = scriptpath.split('/')
    target_index = lan_dic[lang]
    target[0] = target_index

    # ensure that at least 20 char is read
    # no longer seek randomly
    offset = 0
    with open(corpus_dir / scriptpath, 'r', encoding="utf8", errors='ignore') as file:
        file.seek(offset, 0)
        input_char = file.read(input_len)

    # this effectively allows you to skip any non ascii characters
    for idx, char in enumerate(input_char):
        char = char.encode('ascii', errors='ignore').decode('ascii')
        if char not in string_map:
            char = ' '
        if idx == 0:
            pass
        else:
            try:
                bigram_idx=string_map[lastchar]*100+string_map[char]
                inputs[idx, bigram_idx] = 1
            except KeyError:
                char=" "
                print(lastchar)
                bigram_idx=string_map[lastchar]*100+string_map[char]
                inputs[idx, bigram_idx] = 1
        lastchar = char
    bigram_idx = string_map[lastchar] * 100 + string_map[" "]
    inputs[idx, bigram_idx] = 1

    #
    # for idx,char in enumerate(input_char):
    #     try:
    #         char=char.encode('ascii',errors='ignore').decode('ascii')
    #         if char != '':
    #             inputs[idx-empty_count,string_map[char]]=1
    #         else:
    #             empty_count+=1
    #     except IndexError:
    #         print("What?")

    inputs=inputs.sum(0)

    return inputs, target

def training_file_to_bigram(train_valid_point, lan_dic, input_len):
    '''

        :param train_valid_point: (language_script_path, file_len)
        :param lan_dic: dictionary of languages and their target index
        :param input_len: the size of the cut of the file to be fed in
        :return: numpy arrays
        '''

    target = np.zeros(1, dtype=np.long)
    inputs = np.zeros((input_len, 10000))

    scriptpath, file_len = train_valid_point
    lang, file = scriptpath.split('/')
    target_index = lan_dic[lang]
    target[0] = target_index

    # ensure that at least 20 char is read
    # no longer seek randomly
    offset = 0
    with open(corpus_dir / scriptpath, 'r', encoding="utf8", errors='ignore') as file:
        file.seek(offset, 0)
        input_char = file.read(input_len)

    # this effectively allows you to skip any non ascii characters
    for idx, char in enumerate(input_char):
        char = char.encode('ascii', errors='ignore').decode('ascii')
        if char not in string_map:
            char = ' '
        if idx == 0:
            pass
        else:
            try:
                bigram_idx=string_map[lastchar]*100+string_map[char]
                inputs[idx, bigram_idx] = 1
            except KeyError:
                char=" "
                print(lastchar)
                bigram_idx=string_map[lastchar]*100+string_map[char]
                inputs[idx, bigram_idx] = 1
        lastchar = char
    bigram_idx = string_map[lastchar] * 100 + string_map[" "]
    inputs[idx, bigram_idx] = 1

    #
    # for idx,char in enumerate(input_char):
    #     try:
    #         char=char.encode('ascii',errors='ignore').decode('ascii')
    #         if char != '':
    #             inputs[idx-empty_count,string_map[char]]=1
    #         else:
    #             empty_count+=1
    #     except IndexError:
    #         print("What?")

    inputs=inputs.sum(0)

    return inputs, target


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def batch_one_hot_mean(indices,num_classes,batch_size, time_length):
    # the one_hot function above has performance issue, related to generated .eye matrix. O(N^2)
    # indices here is assumed to be word indices of (batch_size, time_length)
    # one_hot will not be necessary for LSTM, as embedding looks up with word indices

    sums=np.zeros((batch_size,num_classes))
    for bidx in range(batch_size):
        np.add.at(sums[bidx,:],indices[bidx,:],1)
    return sums/time_length


def training_file_to_vocab(train_valid_point, lan_dic, vocab_size, lookup, max_len=100, bow=False):
    '''
        1 is all other words
        0 is padding

        :param train_valid_point: (language_script_path, file_len)
        :param lan_dic: dictionary of languages and their target index
        :param vocab_size: the vocabulary size
        :param max_len
        :return: numpy arrays
        '''


    #
    # vocabuary=select_vocabulary(vocab_size-2)
    # # turn vocabulary into a look-up dictionary
    # lookup={}
    # for index, word in enumerate(vocabuary):
    #     lookup[word]=index

    target = np.zeros(1, dtype=np.long)
    inputs = []

    scriptpath, file_len = train_valid_point
    lang, file = scriptpath.split('/')
    target_index = lan_dic[lang]
    target[0] = target_index

    # ensure that at least 20 char is read
    # no longer seek randomly
    with open(corpus_dir / scriptpath, 'r', encoding="utf8", errors='ignore') as file:
        for line in file:
            matches = re.findall(r"[^\W\d_]+|\d+|[^a-zA-Z\d\s:]+", line)
            for match in matches:
                try:
                    inputs.append(lookup[match]+2)
                except KeyError:
                    inputs.append(1)
            if len(inputs)>max_len:
                inputs=inputs[:max_len]
                break
    if len(inputs) == 0:
        inputs.append(0)
    inputs=np.array(inputs,dtype=np.long)

    if bow:
        input_oh=one_hot(inputs,vocab_size)
        if len(input_oh.shape)>1:
            input_oh=input_oh.sum(0)
            input_oh = input_oh / len(inputs)
        else:
            input_oh = np.zeros(vocab_size)
            input_oh[0] = 1
        return input_oh, target
    else:
        return inputs, target


def rep(file):
    # long length and vocab
    # if you don't need it you can chop if off

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

    pkl_dir=corpus_dir.parent/"pkl_dir"

    inputs, target = training_file_to_vocab(file, lan_dic=lan_dic, vocab_size=50000, max_len=1000, bow=False)
    newfilename = file[0] + ".pkl"

    fpath = pkl_dir / newfilename
    if not fpath.is_file():
        with fpath.open("wb") as f:
            pickle.dump((inputs, target), f)

    return 0

def vocab_pickle(file_list, lan_dic, n_proc):
    pkl_dir=corpus_dir.parent/"pkl_dir"
    pkl_dir.mkdir(exist_ok=True)
    for lan in lan_dic:
        land_dir=pkl_dir/lan
        land_dir.mkdir(exist_ok=True)


    # parallel_process(file_list, rep, n_jobs=n_proc)
    Parallel(n_jobs=n_proc)(delayed(rep)(file) for file in file_list)

def pad_sequence(sequences, batch_first=False, padding_value=0):
    r"""Pad a list of variable length Tensors with zero

    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length. For example, if the input is list of
    sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
    otherwise.

    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *`` where `T` is the
            length of the longest sequence.
        Function assumes trailing dimensions and type of all the Tensors
            in sequences are same.

    Arguments:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements. Default: 0.

    Returns:
        Tensor of size ``T x B x *`` if batch_first is False
        Tensor of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor

# def pad_collate_with_factory(args, pad_with=None):
#     val = list(zip(*args))
#     tensors = [[torch.from_numpy(arr) for arr in vv] for vv in val]
#     padded = [pad_sequence(ten,padding_value=pad_with) for ten in tensors]
#     padded[0] = padded[0].permute(1, 0, 2)
#     padded[1] = padded[1].permute(1, 0)
#     return padded

def pad_collate(args):
    """
    :param args:
    :return:
    """
    val=list(zip(*args))
    tensors=[[torch.from_numpy(arr) for arr in vv] for vv in val]
    padded=[pad_sequence(ten) for ten in tensors]
    padded[0]=padded[0].permute(1,0)
    padded[1]=padded[1].permute(1,0)
    return padded

#
# def get_dataloader(ig, vocab_size, *args, **kwargs):
#     dl=DataLoader(ig, collate_fn =pad_collate_with_factory(vocab_size-1),*args,**kwargs)
#     return dl


def pad_collate_2(args):
    """
    :param args:
    :return:
    """
    val=list(zip(*args))
    tensors=[[torch.from_numpy(arr) for arr in vv] for vv in val]
    padded=[pad_sequence(ten) for ten in tensors]
    padded[0]=padded[0].permute(1,0,2)
    padded[1]=padded[1].permute(1,0)
    return padded

#
# def get_dataloader(ig, vocab_size, *args, **kwargs):
#     dl=DataLoader(ig, collate_fn =pad_collate_with_factory(vocab_size-1),*args,**kwargs)
#     return dl

def file_to_array(file, input_len):
    """
    read the first input_len chars
    this method can be improved if you want multiple sampling
    :param file:
    :param input_len:
    :return:
    """
    inputs=np.zeros((input_len,100))

    with open(file,'r', encoding="utf8", errors='ignore') as file:
        input_char=file.read(input_len)

    # this effectively allows you to skip any non ascii characters
    empty_count=0
    for idx,char in enumerate(input_char):
        try:
            char=char.encode('ascii',errors='ignore').decode('ascii')
            if char != '':
                inputs[idx-empty_count,ord(char)]=1
            else:
                empty_count+=1
        except IndexError:
            print("What?")
    return inputs


def file_to_torch(file, input_len):
    array=file_to_array(file,input_len)
    array=torch.from_numpy(array)
    array=array.unsqueeze(0)
    array=array.float().cuda()
    return array

class TimeSeriesIG(Dataset):
    def __init__(self, data, lan_dic, input_len, bigram=False, bow=False):
        self.data=data
        self.lan_dic=lan_dic
        self.input_len=input_len
        self.bigram=bigram
        self.bow=bow

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data_point=self.data[item]
        if self.bigram:
            i,t= training_file_to_bigram(data_point, self.lan_dic, self.input_len)
        elif self.bow:
            i,t= training_file_to_bigram_bow(data_point, self.lan_dic, self.input_len)
        else:
            i,t= training_file_to_input_target_arrays(data_point, self.lan_dic, self.input_len)
        return i,t

class VocabIG(Dataset):
    def __init__(self, file_list, lan_dic, vocab_size, bow=False):
        self.file_list=file_list
        self.lan_dic=lan_dic
        self.vocab_size=vocab_size
        self.bow=bow

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        data_point=self.file_list[item]
        i,t= training_file_to_vocab(data_point, self.lan_dic, self.vocab_size,bow=self.bow)

        return i,t

class VocabIGpkl(Dataset):
    def __init__(self, file_list, lan_dic, vocab_size, max_len, bow=False):
        self.file_list=file_list
        self.lan_dic=lan_dic
        self.vocab_size=vocab_size
        self.max_len=max_len
        self.bow=bow

        if self.vocab_size>50000:
            print("your vocab size is too big")

        if self.max_len>500:
            print('your time length is too big')

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        data_point=self.file_list[item]
        pkl_dir = corpus_dir.parent / "pkl_dir"
        file=data_point[0]
        file=pkl_dir/(file+".pkl")
        with file.open("rb") as f:
            i,t= pickle.load(f)

        # coerce dictionary size, replace all values over vocab_size to be 1, the fill constant
        i[i>self.vocab_size]=1

        # shorten the length if over
        i=i[:self.max_len]

        if self.bow:
            input_oh = one_hot(i, self.vocab_size)
            input_oh = input_oh.sum(0)
            input_oh = input_oh / len(i)
            i,t= input_oh, t.astype(np.long)

            assert(t.shape==(1,))
            try:
                assert(i.shape==(self.vocab_size,))
            except AssertionError:
                assert(i.shape==())
                i=np.zeros(self.vocab_size)
                i[0]=1
            assert(isinstance(i,np.ndarray))
            return i.astype(np.float),t
        else:
            return i, t

class VocabIGBatchpkl():
    def __init__(self, vocab_size, fix_len, bow=False, batch_size=64, valid=False):
        # I do not see a way to do it with Dataset interface. I will have to do it myself.
        self.pkl_dir = pkl_dir
        if valid:
            self.dir = self.pkl_dir / "val"
        else:
            self.dir = self.pkl_dir / "train"

        # no prefetching is implemented at the moment, but I expect the process to go rather fast
        # this is run on a Samsung SSD
        self.num_files=0
        for file in self.dir.iterdir():
            num=int(file.name.split(".")[0])
            if num>self.num_files:
                self.num_files=num

        assert(1024%batch_size==0)
        assert(fix_len <= 1000)
        assert(vocab_size<=50000)
        self.batch_size=batch_size
        self.binb=1024//batch_size


        self.current_batch=None
        self.next_file_index=0
        self.next_batch_idx=0

        self.load_numpy()


        self.vocab_size=vocab_size
        self.max_len=fix_len
        self.bow=bow

    def __len__(self):
        return self.num_files*self.binb

    def load_numpy(self):
        fpath=self.dir/(str(self.next_file_index)+".pkl")
        with fpath.open('rb') as f:
            self.current_batch=pickle.load(f)

    # @staticmethod


    def __next__(self):
        ib,tb=self.current_batch
        i=ib[self.next_batch_idx*self.batch_size:self.next_batch_idx*self.batch_size+self.batch_size]
        t=tb[self.next_batch_idx*self.batch_size:self.next_batch_idx*self.batch_size+self.batch_size]



        # coerce dictionary size, replace all values over vocab_size to be 1, the fill constant
        i[i>self.vocab_size-1]=1

        # shorten the length if over
        i=i[:,:self.max_len]

        if self.bow:
            input_oh = batch_one_hot_mean(i, self.vocab_size, batch_size=self.batch_size, time_length=self.max_len)
            i,t= input_oh, t.astype(np.long)

            assert(t.shape==(self.batch_size,))
            try:
                assert(i.shape==(self.batch_size,self.vocab_size))
            except AssertionError:
                assert(i.shape==(self.batch_size))
                i=np.zeros(self.batch_size, self.vocab_size)
                i[:,0]=1
            assert(isinstance(i,np.ndarray))
            i= i.astype(np.float)
        else:
            i=i.astype(np.long)
        t=t.astype(np.long)

        i, t= Variable(torch.from_numpy(i)), Variable(torch.from_numpy(t))

        # loading finished, house keeping
        self.next_batch_idx+=1

        # if this batch is exhausted
        if self.next_batch_idx>=self.binb:
            # if loading fails, the file points to the next file still.
            self.next_file_index += 1
            self.next_batch_idx=0

            if self.next_file_index>self.num_files:
                raise StopIteration
            self.load_numpy()
        # type type type
        if self.bow:
            return i.float(), t.long()
        else:
            return i.long(), t.long()

    def __iter__(self):
        self.current_batch=None
        self.next_file_index=0
        self.next_batch_idx=0

        self.load_numpy()
        return self

def main():
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
    train, valid = get_data_set(lan_dic)
    ig=TimeSeriesIG(train, lan_dic, 512)
    for i in range(10):
        print(ig[i])

def main2():
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
    train, valid = get_data_set(lan_dic)
    ig=TimeSeriesIG(train, lan_dic, 512,bigram=True)
    for i in range(10):
        a=ig[i]
        print(a)

def main3():
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
    train, valid = get_data_set(lan_dic)
    ig = TimeSeriesIG(train, lan_dic, 512, bow=True)
    for i in range(100):
        a = ig[i]
        print(a)


def main4():
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
    train, valid = get_data_set(lan_dic)
    ig = VocabIG(train, lan_dic,50000)
    for i in range(100):
        a = ig[i]
        print(a)



def rep_batch(files, idx, pkldir, lookup, vocab_size=50000, max_len=1000):
    # long length and vocab
    # if you don't need it you can chop if off

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

    flist=[]
    for file in files:
        inputs, target = training_file_to_vocab(file, lan_dic=lan_dic, vocab_size=vocab_size, lookup=lookup, max_len=max_len, bow=False)
        flist.append((inputs,target))

    # this is the vocab index, by batch and time_len
    input_batch=np.zeros((len(files),max_len), dtype=np.long)
    target_batch=np.zeros((len(files)), dtype=np.long)

    for i in range(len(flist)):
        input, target=flist[i]
        input_batch[i,0:input.shape[0]]=input
        target_batch[i]=target
    newfilename = str(idx) + ".pkl"

    fpath =  pkldir/ newfilename
    if not fpath.is_file():
        with fpath.open("wb") as f:
            pickle.dump((input_batch,target_batch), f)

    return 0

def vocab_batch_pickle(pkl_dir, t, v, batch_size, n_proc, max_len=1000, vocab_size=50000):
    val_dir = pkl_dir/"val"
    train_dir = pkl_dir/"train"
    pkl_dir.mkdir(exist_ok=True)
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    # for lan in lan_dic:
    #     land_dir=train_dir/lan
    #     land_dir.mkdir(exist_ok=True)
    #     land_dir=val_dir/lan
    #     land_dir.mkdir(exist_ok=True)

    # parallel_process(file_list, rep, n_jobs=n_proc)
    vocabuary=select_vocabulary(vocab_size-2)
    # turn vocabulary into a look-up dictionary
    lookup={}
    for index, word in enumerate(vocabuary):
        lookup[word]=index

    for ds, pkl_dir in zip((t,v),(train_dir,val_dir)):
        n_batches=len(t)//batch_size
        # for idx in tqdm(range(n_batches)):
            # rep_batch(ds[idx*batch_size:idx*batch_size+batch_size], lookup=lookup, idx=idx, pkldir=pkl_dir, max_len=max_len, vocab_size=vocab_size)
        Parallel(backend="threading", n_jobs=n_proc)(delayed(rep_batch)(ds[idx*batch_size:idx*batch_size+batch_size], lookup=lookup, idx=idx, pkldir=pkl_dir, max_len=max_len, vocab_size=vocab_size) for idx in range(n_batches))
    print("pickled")

def batch_pickle(n_proc=8, batch_size=256, resplit=True):
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
    train,valid=get_data_set(lan_dic,load=not resplit, save=resplit)
    vocab_batch_pickle(train, valid, batch_size, lan_dic, n_proc)



def main5():
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
    train, valid = get_data_set(lan_dic)
    ig = VocabIGpkl(train, lan_dic, 50000, 100, bow=True)
    ig2 = VocabIG(train, lan_dic,50000, bow=True)
    # for i in range(100):
    #     a = ig[i]
    #     print(a)
    a,b=ig[1714]
    a2,b2=ig2[1714]
    a3,b3=ig[169714]
    a4,b4=ig2[169714]
    assert((a==a2).all())
    assert((b==b2).all())
    assert((a3==a4).all())
    assert((b3==b4).all())
    print("Done")

def testvocabigbatchpkl():
    tig = VocabIGBatchpkl(5000,100,bow=False)

    a = 0
    for i in tig:
        a += 1
        if a < 10:
            print(i)


def main6():
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
    train, valid = get_data_set(lan_dic)
    ig=VocabIGBatchpkl(vocab_size=500, fix_len=100, batch_size=64, bow=False)
    for idx,i in enumerate(ig):
        if idx>100:
            break
        print(i)

    print("Done")

# use this function to repickle.
def repickle(pkl_dir, n_proc=8, batch_size=256):
    resplit=True
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
    train,valid=get_data_set(lan_dic,load=not resplit, save=resplit)
    vocab_batch_pickle(pkl_dir, train, valid, batch_size, lan_dic, n_proc)

    print("Done repickling")


if __name__=="__main__":
    from fileinput import pkl_dir
    main6(pkl_dir)