from scraper.clone_repositories import *

if __name__ == '__main__':
    cm=CloneManager(reset=False)
    cm.main_loop(100000,10)