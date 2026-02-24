VM_NAME=$1
ZONE=$2

if [ -z "$VM_NAME" ] || [ -z "$ZONE" ]; then
  echo "Usage: $0 <vm_name> <zone>"
  exit 1
fi

echo "To kill jobs in: $VM_NAME in $ZONE after 2s..."
sleep 2s


echo 'Killing jobs...'
gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE --worker=all \
    --command "
pgrep -af python | grep 'main.py' | grep -v 'grep' | awk '{print \"sudo kill -9 \" \$1}' | sh
sudo rm -rf /tmp/libtpu_lockfile
" # &> /dev/null
echo 'Killed jobs.'

echo 'Killing carefully'
gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE --worker=all \
    --command "
    ps -eo pid,ppid,stat,cmd | grep 'main.py' | grep -v 'grep' | awk '{print \$1}' | xargs -r sudo kill -9 || true
" # &> /dev/null
echo 'Killed jobs.'

echo 'Killing one by one'
NUM_WORKERS=16

for i in $(seq 0 $((NUM_WORKERS-1))); do
  gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE --worker=$i \
      --ssh-flag="-o ConnectTimeout=1" \
      --ssh-flag="-o ServerAliveInterval=1" \
      --ssh-flag="-o ServerAliveCountMax=1" \
      --command "
      sudo pkill -9 -f python || true
  "  &> /dev/null || true
  echo "Killed jobs on worker $i."
done