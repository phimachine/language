import numpy as np
import os
import random
from torch.utils.data import Dataset, DataLoader
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


def get_data_set(proportion=0.2):
    # open corpus directory
    languages=os.listdir('corpus')
    lan_dic={}
    ct=0
    for lang in languages:
        lan_dic[lang]=ct
        ct+=1

    # split data set to training and validation
    random.seed(1234)
    np.random.seed(1234)

    # train and valid are both lists of (language_script_path, file_length) tuples
    train=[]
    valid=[]
    for language in languages:
        tt,vv=train_valid_files_split(language,proportion)
        train+=tt
        valid+=vv

    # sample minibatch of files

    # get input array and target
    # the input will be a fixed length cut from the file, let's say it's 512 characters, padded if shorter
    # the output will be a one hot array denoting the target
    return train, valid, lan_dic


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
    return train_list,valid_list

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


def file_to_input_target_arrays(train_valid_point, lan_dic, input_len):
    '''

    :param train_valid_point: (language_script_path, file_len)
    :param lan_dic: dictionary of languages and their target index
    :param input_len: the size of the cut of the file to be fed in
    :return:
    '''
    target_dim=len(lan_dic)
    target=np.zeros(target_dim)
    inputs=np.zeros((input_len,256))

    scriptpath, file_len=train_valid_point
    lang,file=scriptpath.split('/')
    target_index=lan_dic[lang]
    target[target_index]=1

    # ensure that at least 20 char is read
    offset=random.randint(0,file_len-20)
    with open('corpus/'+scriptpath,'r') as file:
        file.seek(offset,0)
        input_char=file.read(input_len)
    for idx,char in enumerate(input_char):
        inputs[idx,ord(char)]=1

    return inputs, target

class InputGen(Dataset):
    def __init__(self, data, lan_dic, input_len):
        self.data=data
        self.lan_dic=lan_dic
        self.input_len=input_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data_point=self.data[item]
        i,t= file_to_input_target_arrays(data_point,self.lan_dic,self.input_len)
        return i,t

def main():
    train, valid, lan_dic= get_data_set()
    ig=InputGen(train,lan_dic,512)

if __name__=="__main__":
    main()