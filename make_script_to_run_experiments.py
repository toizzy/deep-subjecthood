import os

results_dir = "latest_run"
curbatch = []
for seed in range(1):
    for sink_idx, sink in enumerate([i.strip() for i in open("sink_langs.txt").readlines()]):
        for source in [i.strip() for i in open("source_langs.txt").readlines()]:
            source = source.strip("-train.conllu")
            if sink.strip('-train.conllu') == source:
                sink_file = sink.strip("-train.conllu") + "-test.conllu"
            else:
                sink_file = sink
            output_fn = os.path.join(results_dir, source.split("/")[-1] + "_" + sink_file.split("/")[-1] + "_" + str(seed))
            if not os.path.exists(os.path.join("results", output_fn)):
                command = f"python run_one_experiment.py --train-lang-base-path {source} --test-lang-fn {sink_file} --only-ao --balance --output-fn {output_fn} --seed {str(seed)}"
                curbatch.append(command)
    print(seed, len(curbatch))
    if len(curbatch) > 0:
        outfile = open(f"run_batch_{seed}.sh", "w")
        for line in curbatch:
            outfile.write(line + "\n")
        outfile.close()
    curbatch = []
