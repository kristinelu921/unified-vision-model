VM_NAME=$1
ZONE=$2

gcloud compute tpus tpu-vm ssh "$VM_NAME" --zone "$ZONE" --worker=all --command "
  cd /mnt/klum/data/mass13k_tfds
  ls
  sudo chmod +777 -R /mnt/klum/data

  cd /kmh-nfs-ssd-us-mount/code/kristine/lvm && python3 -m pip install -r requirements.txt&& python3 one_pass.py
"
