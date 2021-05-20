job = 'source $HOME/.bashrc ; source activate implicit-midpoint-devel ; python main.py --num-samples {} --step-size {} --num-steps {} --thresh {} --integrator {} --hmax {} --{} 2>/dev/null'
with open('joblist.txt', 'w') as f:
    for step_size in [1.0]:
        for num_steps in [1, 2, 5]:
            for seed in range(10):
                for thresh in [1e-3, 1e-6, 1e-9]:
                    for integrator in ['glf', 'imp', 'sglf', 'simp']:
                        for num_samples in [1000]:
                            for hmax in [0.0]:
                                for rs in ['randomize-steps', 'no-randomize-steps']:
                                    f.write(job.format(num_samples, step_size, num_steps, thresh, integrator, hmax, rs) + '\n')
