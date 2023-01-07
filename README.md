# border-legibility

## Computer Vision For International Border Legibility
#### Trevor Ortega, Thomas Nelson, Skyler Crane, Josh Myers-Dean, Scott Wehrwein
Project Page: https://facultyweb.cs.wwu.edu/~wehrwes/BorderLegibility/

## scraping tiles

In order to scrape tiles, you must acquire a key from Bing's API, and put it in "/border-legibility/bing_maps/scrape/bing_key.txt"
This can be acquired at https://www.microsoft.com/en-us/maps/create-a-bing-maps-key

You then will run "/border-legibility/bing_maps/scrape/bing_scrape.py" with (optionally) the 
desired name of the scrape directory as the first command-line argument.

You must then run "/border-legibility/data/getBorders.py" with your scrape directory name as first command line
argument in order to create the .npy shape files for each tile.

**Note: The first couple tiles in each border directory (ie: usa-mex-00000, usa-mex-00001) will likely be unusable, as the border may not extend across the entire tile, since the border ends at this point.** 

## Collecting annotations (Mechanical Turk)

Can collect annotations through the jupyter notebook in the turk directory. Both files have documentation on how to collect annotations. 

If you have never used mturk through jupyter before, here is a post that should help you get set up: https://blog.mturk.com/tutorial-a-beginners-guide-to-crowdsourcing-ml-training-data-with-python-and-mturk-d8df4bdf2977

## Running models

Once all shape files have been created, you can run the various models in "/border-legibility/MODEL/modelComparisons.py" (baseline methods) 
and the siamese model in "/border-legibility/MODEL/siamese.py"

**Note: Running the models requires ground-truth annotations to compare method outputs to.**

#### Siamese:

1. Create image list by running "/border-legibility/bing_maps/create_image_list.py" with the name of the scrape directory as the first command 
line argument. (Ex: $ python3 create_image_list.py global_scrape)

2. Run model by running "/border-legibility/MODEL/siamese.py" with the name of your scrape directory as the first command-line argument.

#### Baselines:

1. In "/border-legibility/MODEL/modelComparisons.py" ensure main method has the correct directory paths;
then, uncomment the method you wish to run and set method parameters if needed.

2. Per-tile results will be saved to "/border-legibility/MODEL/SAVE_DIRECTORY/{method_name}.csv"

3. Method metrics by default will be saved to "/border-legibility/MODEL/pairwise/results/metrics/metrics.csv"
