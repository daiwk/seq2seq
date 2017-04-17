BASE_DIR=`pwd`

export DATA_PATH=${BASE_DIR}/output/

export DATA_TYPE=copy # or reverse

export VOCAB_SOURCE=${DATA_PATH}/nmt_data/toy_${DATA_TYPE}/train/vocab.sources.txt
export VOCAB_TARGET=${DATA_PATH}/nmt_data/toy_${DATA_TYPE}/train/vocab.targets.txt
export TRAIN_SOURCES=${DATA_PATH}/nmt_data/toy_${DATA_TYPE}/train/sources.txt
export TRAIN_TARGETS=${DATA_PATH}/nmt_data/toy_${DATA_TYPE}/train/targets.txt
export DEV_SOURCES=${DATA_PATH}/nmt_data/toy_${DATA_TYPE}/dev/sources.txt
export DEV_TARGETS=${DATA_PATH}/nmt_data/toy_${DATA_TYPE}/dev/targets.txt

export DEV_TARGETS_REF=${DATA_PATH}/nmt_data/toy_${DATA_TYPE}/dev/targets.txt
export TRAIN_STEPS=1000

export MODEL_DIR=${TMPDIR:-/tmp}/nmt_tutorial
mkdir -p $MODEL_DIR


export PRED_DIR=${MODEL_DIR}/pred
mkdir -p ${PRED_DIR}

function train()
{

python -m bin.train \
  --config_paths="
      ./example_configs/nmt_small.yml,
      ./example_configs/train_seq2seq.yml,
      ./example_configs/text_metrics_bpe.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size 32 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR

}

function basic_infer()
{
## basic infer

python -m bin.infer \
  --tasks "
    - class: DecodeText" \
  --model_dir $MODEL_DIR \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $DEV_SOURCES" \
  >  ${PRED_DIR}/predictions.txt.basic
}


function beamsearch_infer()
{
## infer with beamsearch


python -m bin.infer \
  --tasks "
    - class: DecodeText
    - class: DumpBeams
      params:
        file: ${PRED_DIR}/beams.npz" \
  --model_dir $MODEL_DIR \
  --model_params "
    inference.beam_search.beam_width: 5" \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $DEV_SOURCES" \
  > ${PRED_DIR}/predictions.txt.beamsearch
}

function calc_bleu()
{
./bin/tools/multi-bleu.perl ${DEV_TARGETS_REF} < ${PRED_DIR}/predictions.txt.basic
./bin/tools/multi-bleu.perl ${DEV_TARGETS_REF} < ${PRED_DIR}/predictions.txt.beamsearch
}

function main()
{
train
[[ $? -ne 0 ]] && exit 1
basic_infer
[[ $? -ne 0 ]] && exit 1
beamsearch_infer
[[ $? -ne 0 ]] && exit 1
calc_bleu
[[ $? -ne 0 ]] && exit 1

return 0
}
main 2>&1 
