# Neal's Funnel Distribution

Examine the performance of the generalized leapfrog integrator to the implicit midpoint integrator on inference in Neal's funnel distribution. To execute the job, run:
```
dSQ -C cascadelake --jobfile joblist.txt -p day --max-jobs 1000 -c 2 -t 24:00:00 --job-name funnel -o output/funnel-%A-%J.log --suppress-stats-file --submit
```
