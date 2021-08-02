python train.py \
  --mode 'simple_feat_kpcn' \
  --input_channels 34 \
  --hidden_channels 100 \
  --num_layer 9 \
  --eps 0.00316 \
  --do_val \
  --lr 1e-4 \
  --epochs 20 \
  --loss 'L1' \
  --data_dir '/root/kpcn_data/kpcn_data/data' \
  # --do_finetune