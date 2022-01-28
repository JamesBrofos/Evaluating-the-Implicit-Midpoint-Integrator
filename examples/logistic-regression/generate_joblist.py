job = 'source $HOME/.bashrc ; source activate implicit-midpoint-devel ; python main.py --num-samples {} --dataset {} --step-size {} --num-steps {} --integrator {} --{} 2>/dev/null'
datasets = ['diabetis', 'heart', 'breast_cancer']
with open('joblist.txt', 'w') as f:
    for dataset in datasets:
        for step_size in [1e-1, 1e0]:
            for num_steps in [5, 10, 50]:
                for seed in range(10):
                    for num_samples in [10000]:
                        for integrator in ['glf', 'sglf', 'imp', 'simp', 'limp']:
                            for rs in ['randomize-steps', 'no-randomize-steps']:
                                f.write(job.format(num_samples, dataset, step_size, num_steps, integrator, rs) + '\n')
