#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource" -> Ideally want to use gpu, since much faster
#SBATCH --cpus-per-task=8   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham. -> In this case, we're allocating 8 cores
#SBATCH --mem=124G        # memory per node -> lots of memory for our jobs
#SBATCH --time=0-12:00      # time (DD-HH:MM) 12 hours
#SBATCH --output=/home/noelt/scratch/svm.out  # %N for node name, %j for jobID  -> IMPORTANT: needs to be set

module load cuda cudnn # Always need to load this to use TF
source /home/noelt/software/svenv/bin/activate # IMPORTANT: needs to be set

start=`date +%s`

python ./SVCsmall.py

end=`date +%s`
runtime=$((end-start))

printf "\n"
echo "time elapsed (s): $runtime"

