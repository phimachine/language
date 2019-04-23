"""
What does this program do?
It takes the scraped github repository link and language and clones them.
Find the last executed file
"""
import json
from pathlib import Path
import os
import git
from git import Repo
from collections import deque
import pickle
import shutil
import traceback

class LanguageManager():

    def __init__(self, language, maxlen=50):
        self.language=language
        self.dir=self.get_dir()
        self.dir.mkdir(exist_ok=True)
        self.deque=deque(maxlen=maxlen)

    def get_dir(self):
        basedir = Path(os.path.dirname(os.path.realpath(__file__)))
        dir=basedir.parent /"trdata/languages"/self.language
        dir.mkdir(exist_ok=True)
        return dir

    def push(self,git_dir):
        self.deque.appendleft(git_dir)

    def pop(self):
        return self.deque.pop()

    def copy_file(self, file_path):
        file_name=file_path.name
        if not (self.dir/file_name).exists():
            shutil.copyfile(file_path, self.dir/file_name)
            return 1
        else:
            return 0

    def count_files(self):
        return len([file for file in self.dir.iterdir() if file.is_file()])

    def __len__(self):
        return len(self.deque)

    def iter_dir(self):
        return self.dir.iterdir()


class CloneManager():

    def __init__(self, reset=False, json_idx=0, item_idx=0, load=True, save=True):
        self.basedir = Path(os.path.dirname(os.path.realpath(__file__)))
        self.datadir = self.basedir / "items"
        self.configdir=self.basedir/"clone_config"
        self.configdir.mkdir(exist_ok=True)

        # paths to our json files. each contains a git url and its language
        self.datajsons = []
        max_count=0
        for x in self.datadir.iterdir():
            if x.is_file():
                count=int(x.name.split(".")[0])
                if count>max_count:
                    max_count=count

        for i in range(1,max_count+1):
            json_file=self.datadir/(str(i)+".json")
            self.datajsons.append(json_file)
        #
        # for x in self.datadir.iterdir():
        #     if x.is_file():
        #         # this is not ordered by file name. well.
        #         self.datajsons.append(x)

        self.language_list = ["Python", "Go", "Rust", "Java", "Objective-C", "C#", "C++", "C", "Lua", "Ruby", "JavaScript",
                           "Shell"]

        self.save=save
        self.load=load

        self.language_extensions={}
        self.language_extensions["Python"]=[".py"]
        self.language_extensions["Go"]=[".go"]
        self.language_extensions["Rust"]=[".rs",".rlib"]
        self.language_extensions["Java"]=[".java"]
        self.language_extensions["Objective-C"]=[".h",".mm",".m",".M"]
        self.language_extensions["C#"]=[".cs"]
        self.language_extensions["C++"]=[".cpp"]
        self.language_extensions["C"]=[".c"]
        self.language_extensions["Lua"]=[".lua"]
        self.language_extensions["Ruby"]=[".rb"]
        self.language_extensions["JavaScript"]=[".js"]
        self.language_extensions["Shell"]=[".sh",".bash",".zsh"]

        self.repo_dir = self.basedir.parent / "trdata" / "repos"

        self.managers={}
        for language in self.language_list:
            self.managers[language]=LanguageManager(language)

        self.language_file_counts = {}
        for language in self.language_list:
            self.language_file_counts[language] = 0

        # the configs, so when the program terminates you can start again
        self.config={"json_idx":0,
                     "item_idx":0,
                     "language_file_counts":self.language_file_counts}
        if load:
            self.load_config()
            self.load_managers()
        else:
            self.config["json_idx"]=json_idx
            self.config["item_idx"]=item_idx

        # recount the files.
        for language in self.language_list:
            man = self.managers[language]
            count = man.count_files()
            self.language_file_counts[language] = count
        if reset:
            self.config["json_idx"]=0
            self.config["item_idx"]=0

        if save:
            self.save_config()
            self.save_managers()

        # the current converted json items file
        self.item_list=None

    def save_managers(self):
        manager_path=self.configdir/"managers.pkl"
        with manager_path.open('wb') as f:
            pickle.dump(self.managers,f)

    def load_managers(self):
        manager_path=self.configdir/"managers.pkl"
        if manager_path.is_file():
            with manager_path.open('rb') as f:
                self.managers=pickle.load(f)

    def save_config(self):
        config_path=self.configdir/"config.json"
        with config_path.open('w') as f:
            json.dump(self.config, f)

    def load_config(self):
        config_path=self.configdir/"config.json"
        if config_path.is_file():
            with config_path.open('r') as f:
                self.config = json.load(f)
                self.language_file_counts = self.config["language_file_counts"]

    def get_next_item(self):
        try:
            # an item is an iterable of [git_url, repo_language]

            # load the indexed file, starting with the indexed item
            if self.item_list is None:
                try:
                    file_path = self.datajsons[self.config["json_idx"]]
                    with file_path.open('r') as f:
                        self.item_list=json.load(f)
                except IndexError:
                    # this json file does not exist
                    raise FileNotFoundError

            # get the next item from the current list
            try:
                item=self.item_list[self.config["item_idx"]]
            except IndexError:
                # exceeded the length of the list
                self.config["json_idx"]+=1
                self.config["item_idx"]=0
                self.item_list = None
                return self.get_next_item()

            self.config["item_idx"]+=1
            return item

        except FileNotFoundError:
            raise FileNotFoundError("You have finished the whole dataset")


    def clone_item(self,url):
        username, reponame = self.git_url_processor(url)
        repo_name=username+"_"+reponame
        this_repo_path=self.repo_dir/repo_name
        this_repo_path.mkdir(exist_ok=False)
        Repo.clone_from(url, this_repo_path, depth=1)

    def sort_repo(self):
        def onerror(func, path, exc_info):
            """
            Error handler for ``shutil.rmtree``.

            If the error is due to an access error (read only file)
            it attempts to add write permission and then retries.

            If the error is for another reason it re-raises the error.

            Usage : ``shutil.rmtree(path, onerror=onerror)``
            """
            import stat
            if not os.access(path, os.W_OK):
                # Is the error an access error ?
                os.chmod(path, stat.S_IWUSR)
                func(path)
            else:
                raise PermissionError

        # when sort repo begins, the language file counts will be updated continuously
        print("Sorting repo")
        for language in self.language_extensions:
            man=self.managers[language]
            for ext in self.language_extensions[language]:
                for file in self.repo_dir.glob("**/*"+ext):
                    if file.is_file():
                        success=man.copy_file(file)
                        self.language_file_counts[language]+=success
        # then delete the directory.
        shutil.rmtree(self.repo_dir, onerror=onerror)
        self.repo_dir.mkdir(exist_ok=True)

    def git_url_processor(self,url):
        """
        :param url:
        :return: username, reponame
        """
        repo_list=url.split("/")
        return repo_list[-2], repo_list[-1][:-4]


    def main_loop(self, total_clones=100, clones_before_sort=100):
        """

        :param total_clones:
        :param clones_before_sort: save often.
        :return:
        """

        # every total_clones times, a sorting operation is conducted
        last_count=self.language_file_counts.copy()
        try:
            for _ in range(total_clones):
                self.repo_dir.mkdir(exist_ok=True)
                for i in range(clones_before_sort):
                    # which language needs refill?
                    looking_for = min(self.language_file_counts, key=self.language_file_counts.get)
                    look_for_man=self.managers[looking_for]

                    # if the manager is empty
                    if (len(look_for_man)==0):
                        # or else, we need to read items until the desired langauge is found
                        found=False
                        while (not found):

                            git_url, language = self.get_next_item()


                            if language in self.language_list:
                                self.managers[language].push(git_url)
                                if language==looking_for:
                                    found=True
                    git_url = look_for_man.pop()
                    try:
                        self.clone_item(git_url)
                        print("cloned",git_url)
                    except (FileExistsError) as e:
                        print("exists",git_url)
                    except (git.GitCommandError) as e:
                        # cloning something that has been cloned before
                        # pass and when it returns
                        print("git command error")
                self.sort_repo()
                for key,item in self.language_file_counts.items():
                    if self.language_file_counts[key]-last_count[key]>0:
                        print(key,"grew from",last_count[key],"to",self.language_file_counts[key])
                last_count=self.language_file_counts.copy()
                if self.save:
                    self.save_config()
                    self.save_managers()
        except FileNotFoundError:
            print("Whole dataset finished")



def mainrun(newset=False):
    cm=CloneManager(reset=newset)
    cm.main_loop(100000, 10)
    print("Done")


