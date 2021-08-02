python test_gbuffer.py \
  --mode kpcn \
  --diffuse_model 'trained_model/multitask_scale_1/diff_e6.pt' \
  --specular_model 'trained_model/multitask_scale_1/spec_e6.pt' \
  --encoder_model 'trained_model/multitask_scale_1/encode_e6.pt' \
  --albedo_model 'trained_model/multitask_scale_1/albedo_e6.pt' \
  --normal_model 'trained_model/multitask_scale_1/normal_e6.pt' \
  --depth_model 'trained_model/multitask_scale_1/depth_e6.pt' \
  --data_dir '/root/kpcn_data/kpcn_data/data' \
  --save_dir 'test/multitask_scale_1'