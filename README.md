#Deep Subjecthood: Higher Order Grammatical Features in Multilingual BERT

This is the code used to run the experiments in Deep Subjecthood: Higher Order Grammatical Features in Multilingual BERT.

### Use

`run_one_experiment.py` runs one experiment: training a classifier to classify subjects from objects in mBERT embeddings in one training language, and testing this classifier on one test language. Classifiers and mBERT embeddings are cached, to reuse for other experiments with the same train/test language.

`make_script_to_run_all_experiments.py` makes a script to run `run_one_experiment.py` for every pair of languages in `source_langs.txt` and `sink_langs.txt`

To reproduce the experiments in the paper:

1. Change the paths in `source_langs.txt` and `sink_langs.txt`, so that they match the location of the [Universal Dependency Treebanks](https://universaldependencies.org/#download) on your machine
2. Run `make_script_to_run_all_experiments.py` to create the scripts that will run the experiments. This will create a script `run_batch_0.sh` (or more if you're running for multiple seeds
3. Run `run_batch_0.sh`

