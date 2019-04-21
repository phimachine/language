## How to use?

### Scraping with the node.js app

Run `node /scraper/main_scraper.js`. Requires npm install @octokit/rest.

Modify the /scraper/stack file to list the users to start scraping with. The results are periodically saved in 
scraper/items as numbered json files.

It calls the octokit github API to get followers of 



### Sort the scripts

Run `mainrun()` at `/scraper/clone_repositories.py`. It will sort the files you cloned into /trdata directory. It knows to keep
track of all languages, and make the data for each langauge relatively the same.

It will pick up where you left off. Set reset=True to sort from the beginning.

### Serialize the data

Run serialization with `repickle()` at `/fileinput.py`

Data serialization is important, because usually your training files contains thousands of small files.
Your file access will be limited by hard-drive file system access.

You need to setup the serialized data path manually with the repickle() function. Put it in your SSD.

### Train the model

The model is a state of art mixed objective model. It has an accuracy of 96% on balanced validation set at the moment.
It's a mixed objective model modified with Transformer's encoder.

https://www.aaai.org/Papers/AAAI/2019/AAAI-SachanD.7236.pdf

https://arxiv.org/abs/1706.03762

Select the model that you to train from the `/train.py`. The newest mdoel is `mixed_bow_transformer_cache()`.