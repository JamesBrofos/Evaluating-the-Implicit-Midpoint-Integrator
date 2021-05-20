job = 'source $HOME/.bashrc ; source activate implicit-midpoint-devel ; python main.py --num-samples {} --step-size {} --num-steps {} --thresh {} --{} 2>/dev/null'
with open('joblist.txt', 'w') as f:
    for num_steps in [20]:
        for step_size in [0.1, 0.2, 0.5]:
            for seed in range(10):
                for num_samples in [10000]:
                    for thresh in [1e-3, 1e-6, 1e-9]:
                        for rs in ['randomize-steps', 'no-randomize-steps']:
                            f.write(job.format(num_samples, step_size, num_steps, thresh, rs) + '\n')
