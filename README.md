## Basic Example for Ray [Tune] on Slurm. 

Just replace the simple training function with a more complicated one, 
taking the parameters to optimize as an input dict. 
If parameters should be passed to the function, that should not be optimized, use functools.Partial


I had a bit of trouble setting up ray, so maybe this helps if you run into trouble on max: 
module load maxwell
module load gcc
module load anaconda/5.2

conda create -n "ray_env" --clone pytorch1.9
conda activate ray_env
pip install ray[tune]

sbatch submit.sh

This should work out of the box
