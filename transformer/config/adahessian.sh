DATA_PATH=./data-bin/iwslt14.tokenized.de-en.joined
model=transformer
PROBLEM=iwslt14_de_en
ARCH=transformer_iwslt_de_en_v2
OUTPUT_PATH=log/adahessian
NUM=5
lr=0.047
# warmup-init-lr: using lr / adam_lr * adam_warmup-lr
wlr=3.13333333e-6 
# similar as above
mlr=3.13333333e-8
mkdir -p $OUTPUT_PATH

export CUDA_VISIBLE_DEVICES=1; python train.py ${DATA_PATH} \
                            --seed 1 \
                            --adam-eps 1e-04 \
                            --single-gpu \
                            --hessian-power 1 \
                            --arch ${ARCH} --share-all-embeddings \
                            --optimizer adahessian --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
                            --block-length 32 \
                            --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 \
                            --criterion label_smoothed_cross_entropy \
                            --lr-scheduler inverse_sqrt --warmup-init-lr ${wlr} --warmup-updates 8000 \
                            --lr ${lr} --min-lr ${mlr} \
                            --label-smoothing 0.1 --weight-decay 0.0001 \
                            --max-tokens 4096 --save-dir ${OUTPUT_PATH} \
                            --update-freq 1 --no-progress-bar --log-interval 50 \
                            --ddp-backend no_c10d \
                            --keep-last-epochs ${NUM} --early-stop ${NUM} \
                            --restore-file ${OUTPUT_PATH}/checkpoint_best.pt \
                            | tee -a ${OUTPUT_PATH}/train_log.txt

python scripts/average_checkpoints.py --inputs ${OUTPUT_PATH} --num-epoch-checkpoints ${NUM} --output ${OUTPUT_PATH}/averaged_model.pt

BEAM_SIZE=5
LPEN=1.0
TRANS_PATH=${OUTPUT_PATH}/trans
RESULT_PATH=${TRANS_PATH}/

mkdir -p $RESULT_PATH
CKPT=averaged_model.pt

export CUDA_VISIBLE_DEVICES=1; python generate.py \
    ${DATA_PATH} \
    --path ${OUTPUT_PATH}/${CKPT} \
    --batch-size 128 \
    --beam ${BEAM_SIZE} \
    --lenpen ${LPEN} \
    --remove-bpe \
    --log-format simple \
    --source-lang de \
    --target-lang en \
> ${RESULT_PATH}/res.txt