# Logistic Regression

Examine the performance of the generalized leapfrog integrator to the implicit midpoint integrator on Bayesian logistic regression. To execute the job, run:
```
mkdir output
dSQ -C cascadelake --jobfile joblist.txt -p day --max-jobs 1000 -c 5 -t 24:00:00 --job-name logistic -o output/logistic-%A-%J.log --suppress-stats-file --submit
```

Individual datasets have not been included in this repository.
