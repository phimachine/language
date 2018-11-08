import numpy as np
import os
import random
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch
import sys

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


def get_data_set(lan_dic,proportion=0.2):

    # split data set to training and validation
    random.seed(1234)
    np.random.seed(1234)

    # train and valid are both lists of (language_script_path, file_length) tuples
    train=[]
    valid=[]
    for language in lan_dic:
        tt,vv=train_valid_files_split(language,proportion)
        train+=tt
        valid+=vv

    # sample minibatch of files

    # get input array and target
    # the input will be a fixed length cut from the file, let's say it's 512 characters, padded if shorter
    # the output will be a one hot array denoting the target
    return train, valid


def train_valid_files_split(language, proportion=0.2):
    """
    Given a language folder, splits to train valid, calculates the file length

    :param language: name of the language
    :param proportion: proportion of validation set
    :return:
    """
    all_files=os.listdir("corpus/"+language)
    valid_size=int(len(all_files)*proportion)+1
    valid_list=random.sample(all_files,valid_size)
    train_list=[]
    for file in all_files:
        if file not in valid_list:
            language_script_path=language+'/'+file
            train_list.append((language_script_path,get_file_length(language_script_path)))
    for i in range(len(valid_list)):
        lsp=language+'/'+valid_list[i]
        valid_list[i]=(lsp,get_file_length(lsp))

    new_train_list=[]
    new_valid_list=[]

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
        with open("corpus/"+language_script_path, 'r', encoding="utf-8") as file:
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
    inputs=np.zeros((input_len,256))

    scriptpath, file_len=train_valid_point
    lang,file=scriptpath.split('/')
    target_index=lan_dic[lang]
    target[0]=target_index

    # ensure that at least 20 char is read
    offset=random.randint(0,file_len-20)
    with open('corpus/'+scriptpath,'r', encoding="utf8", errors='ignore') as file:
        file.seek(offset,0)
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

    return inputs, target

def file_to_array(file, input_len):
    """
    read the first input_len chars
    this method can be improved if you want multiple sampling
    :param file:
    :param input_len:
    :return:
    """
    inputs=np.zeros((input_len,256))

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

class InputGen(Dataset):
    def __init__(self, data, lan_dic, input_len):
        self.data=data
        self.lan_dic=lan_dic
        self.input_len=input_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data_point=self.data[item]
        i,t= training_file_to_input_target_arrays(data_point, self.lan_dic, self.input_len)
        return i,t

def main():
    train, valid, lan_dic= get_data_set()
    ig=InputGen(train,lan_dic,512)

if __name__=="__main__":
    main()