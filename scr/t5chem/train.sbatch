#!/bin/bash

#SBATCH --account=torch_pr_149_chemistry
#SBATCH --job-name=t5chem
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=4:00:00
#SBATCH --mem=16GB

Model_dir=/scratch/yw5806/Multistep_final/FinetunedBC_models/FinetunedBC_2/
Data_dir=/scratch/yw5806/Multistep_final/data/splitting_2/concat+0yield/

cd /scratch/yw5806/yw5806_transfer_backup/environments/
if [[ $(hostname -s) =~ ^g ]]; then nv="--nv"; fi

singularity exec $nv \
	    --overlay t5chem.ext3:ro \
	    /scratch/yw5806/yw5806_transfer_backup/environments/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
	    /bin/bash -c "
        source /ext3/env.sh

		cd $Model_dir
		python Multistep_vers3.py --data_dir $Data_dir --model_dir $Model_dir \
		--pretrain /scratch/yw5806/Multistep_final/ModelBC/Model4/best_cp-18228/

		echo '-------------- Predicting multistep test sets ---------------'
		python Multistep_test.py\
		--data_dir $Data_dir \
		--model_dir $Model_dir \
		--output_name "predictions_multistep.csv"
		
		
		"

#cd $Model_dir
#sbatch $Model_dir/predict.sbatch

echo done
