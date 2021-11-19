#!/bin/bash
#SBATCH --partition=allgpu                  #choose the partition on which to run
#SBATCH --constraint='V100'                 #choose constraint for gpus -> important otherwise pytorch is not compatible
#SBATCH --time=00:30:00                           # Maximum time requested
#SBATCH --nodes=2                                # Number of nodes, note that these are the different maxwell machines!
#SBATCH --chdir=/home/kaechben/slurm/output        # directory must already exist!, here all of the 
#SBATCH --job-name=hostname
#SBATCH --output=%j.out               # File to which STDOUT will be written
#SBATCH --error=%j.err                # File to which STDERR will be written
#SBATCH --mail-type=END                           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=max.muster@desy.de            # Email to which notifications will be sent. It defaults to <userid@mail.desy.de> if none is set.
unset LD_PRELOAD
source /etc/profile.d/modules.sh
module purge
module load maxwell gcc/9.3
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

  
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
   --block &

# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
         --block &
    sleep 5
done


python -u ~/raytunes/sample_train.py "$SLURM_CPUS_PER_TASK"
