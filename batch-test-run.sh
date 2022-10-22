python train_BSS_distillation.py --compress 0 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 1 --t_net_id ResNet26 --s_net_id ResNet8  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 2  | tee -a batch-test-results.csv &
python train_BSS_distillation.py --compress 1 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 1 --t_net_id ResNet26 --s_net_id ResNet8  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 3  | tee -a batch-test-results.csv &
python train_BSS_distillation.py --compress 0 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 3 --t_net_id ResNet26 --s_net_id ResNet8  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 1  | tee -a batch-test-results.csv &
wait
python train_BSS_distillation.py --compress 1 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 3 --t_net_id ResNet26 --s_net_id ResNet8  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 2  | tee -a batch-test-results.csv &
python train_BSS_distillation.py --compress 0 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 7 --t_net_id ResNet26 --s_net_id ResNet8  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 3  | tee -a batch-test-results.csv &
python train_BSS_distillation.py --compress 1 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 7 --t_net_id ResNet26 --s_net_id ResNet8  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 1  | tee -a batch-test-results.csv &
wait
python train_BSS_distillation.py --compress 0 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 9 --t_net_id ResNet26 --s_net_id ResNet8  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 2  | tee -a batch-test-results.csv &
python train_BSS_distillation.py --compress 1 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 9 --t_net_id ResNet26 --s_net_id ResNet8  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 3  | tee -a batch-test-results.csv &
python train_BSS_distillation.py --compress 0 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 1 --t_net_id ResNet26 --s_net_id ResNet14  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 1  | tee -a batch-test-results.csv &
wait
python train_BSS_distillation.py --compress 1 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 1 --t_net_id ResNet26 --s_net_id ResNet14  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 2  | tee -a batch-test-results.csv &
python train_BSS_distillation.py --compress 0 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 3 --t_net_id ResNet26 --s_net_id ResNet14  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 3  | tee -a batch-test-results.csv &
python train_BSS_distillation.py --compress 1 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 3 --t_net_id ResNet26 --s_net_id ResNet14  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 1  | tee -a batch-test-results.csv &
wait
python train_BSS_distillation.py --compress 0 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 7 --t_net_id ResNet26 --s_net_id ResNet14  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 2  | tee -a batch-test-results.csv &
python train_BSS_distillation.py --compress 1 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 7 --t_net_id ResNet26 --s_net_id ResNet14  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 3  | tee -a batch-test-results.csv &
python train_BSS_distillation.py --compress 0 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 9 --t_net_id ResNet26 --s_net_id ResNet14  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 1  | tee -a batch-test-results.csv &
wait
python train_BSS_distillation.py --compress 1 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 9 --t_net_id ResNet26 --s_net_id ResNet14  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 2  | tee -a batch-test-results.csv &
python train_BSS_distillation.py --compress 0 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 1 --t_net_id ResNet26 --s_net_id ResNet20  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 3  | tee -a batch-test-results.csv &
python train_BSS_distillation.py --compress 1 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 1 --t_net_id ResNet26 --s_net_id ResNet20  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 1  | tee -a batch-test-results.csv &
wait
python train_BSS_distillation.py --compress 0 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 3 --t_net_id ResNet26 --s_net_id ResNet20  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 2  | tee -a batch-test-results.csv &
python train_BSS_distillation.py --compress 1 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 3 --t_net_id ResNet26 --s_net_id ResNet20  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 3  | tee -a batch-test-results.csv &
python train_BSS_distillation.py --compress 0 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 7 --t_net_id ResNet26 --s_net_id ResNet20  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 1  | tee -a batch-test-results.csv &
wait
python train_BSS_distillation.py --compress 1 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 7 --t_net_id ResNet26 --s_net_id ResNet20  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 2  | tee -a batch-test-results.csv &
python train_BSS_distillation.py --compress 0 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 9 --t_net_id ResNet26 --s_net_id ResNet20  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 3  | tee -a batch-test-results.csv &
python train_BSS_distillation.py --compress 1 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 9 --t_net_id ResNet26 --s_net_id ResNet20  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 1  | tee -a batch-test-results.csv &
wait
python train_BSS_distillation.py --compress 0 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 1 --t_net_id ResNet26 --s_net_id ResNet26  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 2  | tee -a batch-test-results.csv &
python train_BSS_distillation.py --compress 1 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 1 --t_net_id ResNet26 --s_net_id ResNet26  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 3  | tee -a batch-test-results.csv &
python train_BSS_distillation.py --compress 0 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 3 --t_net_id ResNet26 --s_net_id ResNet26  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 1  | tee -a batch-test-results.csv &
wait
python train_BSS_distillation.py --compress 1 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 3 --t_net_id ResNet26 --s_net_id ResNet26  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 2  | tee -a batch-test-results.csv &
python train_BSS_distillation.py --compress 0 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 7 --t_net_id ResNet26 --s_net_id ResNet26  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 3  | tee -a batch-test-results.csv &
python train_BSS_distillation.py --compress 1 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 7 --t_net_id ResNet26 --s_net_id ResNet26  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 1  | tee -a batch-test-results.csv &
wait
python train_BSS_distillation.py --compress 0 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 9 --t_net_id ResNet26 --s_net_id ResNet26  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 2  | tee -a batch-test-results.csv &
python train_BSS_distillation.py --compress 1 --dataset CIFAR-10 --attack_id BSS --attack_size 64 --temp 9 --t_net_id ResNet26 --s_net_id ResNet26  --max_epoch 10 --save_dir results/BSS_distillation_80epoch_res8_C10 --gpu 3  | tee -a batch-test-results.csv &
