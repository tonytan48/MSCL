#export USE_ADAPTER=PET
#export ADAPTER_H=384
for rand_seed in 99 10 11 10; do
python moco_domain.py \
  --source_domain 'dvd kitchen electronics'\
  --model_name 'roberta-large'\
  --embedding_size 1024\
  --lr 1e-5 \
  --nepoch 10\
  --seed $rand_seed\
  --lambda_supcon 0.0\
  --batch_size 16\
  --grad_clip_norm 10.0\
  --gradient_acc_step 1\
  --memory_bank_size 128\
  --skip_step 0\
  --warmup_step 200\
  --lambda_ce 1.0\
  --lambda_supcon 0.0\
  --m_update_interval 20\
  --lambda_adv 0.0\
  --lambda_moco 1.0\
  --hidden_dropout_prob 0.1\
  --hidden_size 256\
  --temp 0.2\
  --save_model True\


done;
