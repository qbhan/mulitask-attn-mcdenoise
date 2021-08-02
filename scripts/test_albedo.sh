python test_albedo.py \
  --mode kpcn \
  --diffuse_model 'trained_model/multitask_albedo_denoise_finetune_6_11/diff_e13.pt' \
  --specular_model 'trained_model/multitask_albedo_denoise_finetune_6_11/spec_e13.pt' \
  --encoder_model 'trained_model/multitask_albedo_denoise_finetune_6_11/encode_e13.pt' \
  --albedo_model 'trained_model/multitask_albedo_denoise_finetune_6_11/albedo_e13.pt' \
  --gradX_model 'trained_model/multitask_albedo_grad_scale_1_denoise_finetune_4/gradX_e3.pt' \
  --gradY_model 'trained_model/multitask_albedo_grad_scale_1_denoise_finetune_4/gradY_e3.pt' \
  --data_dir '/root/kpcn_data/kpcn_data/data' \
  --save_dir 'test/multitask_albedo_denoise_finetune_6_11'