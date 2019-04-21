from pathlib import Path
import shutil

root_dir=Path(".")
max_count=-1
for json in (root_dir/"items").iterdir():
    fname=json.name
    count=fname.split(".")[0]
    count=int(count)
    if count>max_count:
        max_count=count

max_count+=1
for set in (root_dir/"old_items").iterdir():
    for json in set.iterdir():
        shutil.copyfile(json,root_dir/"items"/(str(max_count)+".json"))
        max_count+=1
