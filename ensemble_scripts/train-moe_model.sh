conf=$1

DEFAULT_GPU_ID=0
if [ -z ${CUDA_VISIBLE_DEVICES+x} ]; then
  GPU_ID=$DEFAULT_GPU_ID
  echo "set CUDA_VISIBLE_DEVICES to default('$GPU_ID')"
else
  GPU_ID=$CUDA_VISIBLE_DEVICES
  echo "set CUDA_VISIBLE_DEVICES to external('$GPU_ID')"
fi

train_path=/home/lulu/yt8m/models/model_predictions
train_data_patterns=""
for d in $(cat $conf); do
  train_data_patterns="${train_path}/${d}/*.tfrecord${train_data_patterns:+,$train_data_patterns}"
done
echo $train_data_patterns

CUDA_VISIBLE_DEVICES="$GPU_ID" python train.py \
      --train_dir="/home/lulu/yt8m/models/ensamble/MoeModel" \
      --train_data_patterns="$train_data_patterns" \
      --model=MoeModel \
      --keep_checkpoint_every_n_hours=0.1 \
      --batch_size=32 \
      --num_epochs=5
