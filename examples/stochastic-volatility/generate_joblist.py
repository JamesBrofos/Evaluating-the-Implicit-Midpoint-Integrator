job = 'source $HOME/.bashrc ; source activate implicit-midpoint-devel ; python main.py --num-burn {} --num-samples {} --integrator {} --{} 2>/dev/null'
with open('joblist.txt', 'w') as f:
    for integrator in ['glf', 'sglf', 'imp', 'simp', 'limp']:
        for num_burn in [10000]:
            for num_samples in [20000]:
                for seed in range(100):
                    for rs in ['randomize-steps', 'no-randomize-steps']:
                        f.write(job.format(num_burn, num_samples, integrator, rs) + '\n')
