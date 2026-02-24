VM_NAME=$1
ZONE=$2

gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
--worker=all --command "
mkdir -p ~/bin
cat << 'EOF' > ~/bin/mark_last_cmd.sh
#!/usr/bin/env bash
set -euo pipefail
cmd=\"\$*\"
if [ -z \"\$cmd\" ]; then
  echo \"usage: \$0 \\\"command\\\"\" 1>&2
  exit 1
fi
ts=\$(date -u +\"%Y-%m-%dT%H:%M:%SZ\")
host=\$(hostname)
printf '{\"ts\":\"%s\",\"host\":\"%s\",\"cmd\":%s}\\n' \"\$ts\" \"\$host\" \"\$(printf '%s' \"\$cmd\" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')\" > ~/.last_cmd
EOF
chmod +x ~/bin/mark_last_cmd.sh

~/bin/mark_last_cmd.sh \"init: pip installs\"
python3 -m pip install absl-py==1.4.0
python3 -m pip install clu==0.0.11
python3 -m pip install -U pip setuptools wheel
python3 -m pip install flax==0.10.5
python3 -m pip install -U 'jax[tpu]==0.6.2' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python3 -m pip install numpy==1.26.4
python3 -m pip install tensorflow==2.16.1
python3 -m pip install torch==2.2.2
python3 -m pip install torchvision==0.17.2
python3 -m pip install chex==0.1.86
python3 -m pip install wandb
python3 -m pip install tqdm
python3 -m pip install optax==0.2.1
python3 -m pip install ml_dtypes==0.5.3
python3 -m pip install orbax-checkpoint==0.11.25
python3 -m pip install tensorstore==0.1.78
python3 -m pip install datasets
python3 -m pip install webdataset
python3 -m pip install huggingface_hub
python3 -m pip install 'huggingface_hub[cli,torch]'
python3 -m pip install 'transformers<5.0.0'
python3 -m pip install hf_transfer
python3 -m pip install gcsfs
python3 -m pip install tmpfs
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=1 #TODO: add your own token
"


# sanity check
gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
--worker=all --command "
~/bin/mark_last_cmd.sh \"init: sanity check\"
export JAX_PLATFORMS=tpu
python3 -c 'import jax; print(jax.device_count())'
python3 -c 'import jaxlib; print(jaxlib.__version__)'
export WANDB_API_KEY=1 #TODO: add your own api key
export WANDB_ENTITY=1 #TODO: add your own entity
export WANDB_PROJECT=1 #TODO: add your own project name

"

# # mount NFS Filestore
gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
--worker=all --command "

~/bin/mark_last_cmd.sh \"init: mount nfs + install requirements\"
sudo apt-get -y update
sudo apt-get -y install nfs-common

sudo mkdir -p /kmh-nfs-ssd-us-mount
sudo mount -o vers=3 10.97.81.98:/kmh_nfs_ssd_us /kmh-nfs-ssd-us-mount
sudo chmod go+rw /kmh-nfs-ssd-us-mount
cd /kmh-nfs-ssd-us-mount/code/kristine/kristine-jit
python3 -m pip install -r requirements.txt
cd /kmh-nfs-ssd-us-mount/code/kristine/lvm
python3 -m pip install -r requirements.txt
ls /kmh-nfs-ssd-us-mount
"