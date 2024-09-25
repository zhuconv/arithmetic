
JOBNAME=$(cat debugpy-jobid)
echo "Getting host for Slurm job $JOBNAME"

trap "scancel -n $JOBNAME; exit 0" EXIT
# Start TCP proxy from compute node
SLURM_COMPUTE_VM=$(squeue -u $USER --name=$JOBNAME --states=R -h -O NodeList | xargs)
echo "Starting proxy from ${SLURM_COMPUTE_VM}:3000 to 127.0.0.1:$1"
ssh -L $1:127.0.0.1:3000 ${SLURM_COMPUTE_VM}
# echo "socat"
# socat tcp-listen:$1,bind=127.0.0.1,forever,interval=10,fork tcp:${SLURM_COMPUTE_VM}:3000