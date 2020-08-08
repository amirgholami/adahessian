DATA_PATH=./data-bin/iwslt14.tokenized.de-en.joined
BEAM_SIZE=5
LPEN=1.0
OUTPUT_PATH=pretrained_result
TRANS_PATH=${OUTPUT_PATH}/trans
RESULT_PATH=${TRANS_PATH}/

mkdir -p $RESULT_PATH
CKPT=averaged_model.pt

export CUDA_VISIBLE_DEVICES=1; python generate.py \
    ${DATA_PATH} \
    --path adahessian_pretrained_model.pt \
    --batch-size 128 \
    --beam ${BEAM_SIZE} \
    --lenpen ${LPEN} \
    --remove-bpe \
    --log-format simple \
    --source-lang de \
    --target-lang en \
> ${RESULT_PATH}/res.txt