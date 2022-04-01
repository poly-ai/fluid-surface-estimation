# setup virtual env
python3 -m venv env

# enter venv
source "env/bin/activate"

# install dependencies
python3 -m pip install -r "requirements.txt"

# then run "sbatch run.sh" to queue a batch job
# squeue --user=$USER to view your jobs
