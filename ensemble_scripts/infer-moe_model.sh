model=MoeModel
conf=$1

DEFAULT_GPU_ID=0
if [ -z ${CUDA_VISIBLE_DEVICES+x} ]; then
  GPU_ID=$DEFAULT_GPU_ID
  echo "set CUDA_VISIBLE_DEVICES to default('$GPU_ID')"
else
  GPU_ID=$CUDA_VISIBLE_DEVICES
  echo "set CUDA_VISIBLE_DEVICES to external('$GPU_ID')"
fi

test_path=/home/lulu/yt8m/models/model_predictions_test
test_data_patterns=""
for d in $(cat $conf); do
  test_data_patterns="${test_path}/${d}/*.tfrecord${test_data_patterns:+,$test_data_patterns}"
done
echo "$test_data_patterns"

CUDA_VISIBLE_DEVICES="$GPU_ID" python inference.py \
      --model_checkpoint_path="/home/lulu/yt8m/models/ensamble/${model}/model.ckpt-2503" \
      --output_file="/home/lulu/yt8m/models/ensamble_inferece/${model}/predictions.${model}.csv" \
      --model="MoeModel" \
      --input_data_patterns="$test_data_patterns" \
      --segment_labels \
      --top_k=1000
