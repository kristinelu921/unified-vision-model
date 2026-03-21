BUCKET_ZONE=$1 #eg: us-central1-b
LOCAL_PATH=$2 #eg: /kmh-nfs-ssd-us-mount/data/kristine/lvm/mass13k_latents_tfds
BUCKET_DEST=$3 #eg: kristine/lvm/mass13k_latents_tfds

#ex command: ./store_data_to_buckets.sh us-central1-b /kmh-nfs-ssd-us-mount/data/kristine/lvm/mass13k_latents_tfds kristine/lvm/mass13k_latents_tfds

# bucket maps:
if [ "$BUCKET_ZONE" == "asia-northeast1-b" ]; then
    BUCKET_DEST="gs://kmh-gcp-asia-northeast1-b/"+$BUCKET_DEST #eg: gs://kmh-gcp-asia-northeast1-b/kristine/lvm/mass13k_latent_tfds
elif [ "$BUCKET_ZONE" == "us-central1" ]; then
    BUCKET_DEST="gs://kmh-gcp-us-central1/"+$BUCKET_DEST
elif [ "$BUCKET_ZONE" == "us-central2" ]; then
    BUCKET_DEST="gs://kmh-gcp-us-central2/"+$BUCKET_DEST
elif [ "$BUCKET_ZONE" == "us-east1" ]; then
    BUCKET_DEST="gs://kmh-gcp-us-east1/"+$BUCKET_DEST
elif [ "$BUCKET_ZONE" == "us-east5" ]; then
    BUCKET_DEST="gs://kmh-gcp-us-east5/"+$BUCKET_DEST
elif [ "$BUCKET_ZONE" == "us-west4" ]; then
    BUCKET_DEST="gs://kmh-gcp-us-west4/"+$BUCKET_DEST
else
    echo "Invalid bucket zone"
    exit 1
fi

echo "Copying files from $LOCAL_PATH to $BUCKET_DEST"
ls -la "$LOCAL_PATH"

gcloud storage cp -r --no-clobber "$LOCAL_PATH" "$BUCKET_DEST"

echo "Listing files in $BUCKET_DEST"
gcloud storage ls "$BUCKET_DEST"/*

