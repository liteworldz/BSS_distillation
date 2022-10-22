for s_net_id in ResNet8 ResNet14 ResNet20 ResNet26; do
	for temp in 1 3 7; do
		eval $(echo "python train_BSS_distillation.py --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp $temp --t_net_id ResNet26 --s_net_id $s_net_id  --gpu 0 --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 ")  | tee -a BSS-output.csv
	done
done