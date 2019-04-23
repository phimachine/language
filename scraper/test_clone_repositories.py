import unittest
from scraper.clone_repositories import *
from scraper.clone_repositories import CloneManager, LanguageManager

class CloneRepoTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cm=CloneManager()

    def test_language_manager_init(self):
        lm = LanguageManager("Python")
        self.assertIsInstance(lm, LanguageManager, "Initialization test")

    def test_clone_manager_init(self):
        self.assertIsInstance(self.cm, CloneManager, "Initialization test")
        self.assertIsInstance(self.cm.basedir, Path)
        self.assertIsInstance(self.cm.configdir, Path)

    def test_get_next_item(self):
        try:
            item=self.cm.get_next_item()
            self.assertTrue(".git" in item[0], "should be a git link")
        except Exception as e:
            self.assertIsInstance(e, FileNotFoundError, "All files exhausted")

    def test_clone_item(self):
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

        url='https://github.com/splashofcrimson/shh.git'

        username, reponame = self.cm.git_url_processor(url)
        repo_name=username+"_"+reponame
        this_repo_path=self.cm.repo_dir/repo_name
        if this_repo_path.exists():
            shutil.rmtree(this_repo_path, onerror=onerror)
        self.cm.clone_item(url)
        self.assertTrue(this_repo_path.exists(), "directory should be created for cloned repo")
        self.assertTrue(len(list(this_repo_path.iterdir()))>0,"cloned repo should have some files")
        shutil.rmtree(this_repo_path, onerror=onerror)


    def test_sort_repo(self):
        old_count=self.cm.language_file_counts
        self.cm.sort_repo()
        self.assertTrue(len(list(self.cm.repo_dir.iterdir()))==0,"repo dir should be sorted and cleaned")
        new_count=self.cm.language_file_counts
        for lan, count in old_count.items():
            self.assertTrue(new_count[lan]>=count, "sorting should not reduce the file counts")
            man=self.cm.managers[lan]
            second_count=man.count_files()
            self.assertEqual(second_count,new_count[lan],"file count incorrectly accumulated")

    def test_main_loop(self):
        try:
            self.cm.main_loop(2,2)
        except Exception as e:
            self.assertIsInstance(e, FileNotFoundError, "The only exception should be FileNotFoundError")


if __name__ == '__main__':
    unittest.main()