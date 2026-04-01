#!/bin/bash
#SBATCH --array=1-1000%50
#SBATCH --partition=issa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Calculate which lines this specific task will execute
# Each of the 1000 tasks will do ~xx lines of work


TASK_LIST="all_tasks.txt"
TOTAL_LINES=$(wc -l < "$TASK_LIST")

LINES_PER_TASK=$((($TOTAL_LINES +999) / 1000)) # ceiling division so nothing gets missed
START_LINE=$(( ($SLURM_ARRAY_TASK_ID - 1) * $LINES_PER_TASK + 1 ))
END_LINE=$(( $START_LINE + $LINES_PER_TASK - 1 ))

sed -n "${START_LINE},${END_LINE}p" "$TASK_LIST" | while read -r LINE; do 
	if [  -n "$LINE" ]; then
		bash $LINE
	fi
done
