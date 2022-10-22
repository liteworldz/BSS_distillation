'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
from argparse import ArgumentParser


import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import attacks

from models import *


import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2'

# Training
def train_attack_KD(use_cuda, compress, optimizer, attack_id, t_net_id, s_net_id, attack, trainloader, criterion_CE, temperature, attack_size, t_net, s_net, ratio, ratio_attack, epoch):
    epoch_start_time = time.time()
    
    s_net.train()
    t_net.eval()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        batch_size1 = inputs.shape[0]

        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)

        out_s = s_net(inputs)

        # Cross-entropy loss
        loss = criterion_CE(out_s[0:batch_size1, :], targets)
        out_t = t_net(inputs)
        
        '''''
        print("out_t ", out_t.shape)
        print("out_s ", out_s.shape)
         ([256, 10])
        '''''
        
        ## Determine Total variance in the space 
        
        # KD loss
        
        if compress:
            a = out_t.detach().cpu().numpy()
            ua,sa,vha=np.linalg.svd(a.T,full_matrices=False)
            loss += - ratio * (F.softmax(torch.tensor(vha.T).cuda(), 1).detach() * F.log_softmax(out_s/temperature, 1)).sum() / batch_size1
        else:    
            loss += - ratio * (F.softmax(out_t/temperature, 1).detach() * F.log_softmax(out_s/temperature, 1)).sum() / batch_size1
        
        
        if ratio_attack > 0:

            condition1 = targets.data == out_t.sort(dim=1, descending=True)[1][:, 0].data
            condition2 = targets.data == out_s.sort(dim=1, descending=True)[1][:, 0].data

            attack_flag = condition1 & condition2

            if attack_flag.sum():
                # Base sample selection
                attack_idx = attack_flag.nonzero().squeeze()
                if attack_idx.shape[0] > attack_size:
                    diff = (F.softmax(out_t[attack_idx,:], 1).data - F.softmax(out_s[attack_idx,:], 1).data) ** 2
                    distill_score = diff.sum(dim=1) - diff.gather(1, targets[attack_idx].data.unsqueeze(1)).squeeze()
                    attack_idx = attack_idx[distill_score.sort(descending=True)[1][:attack_size]]

                # Target class sampling
                attack_class = out_t.sort(dim=1, descending=True)[1][:, 1][attack_idx].data
                class_score, class_idx = F.softmax(out_t, 1)[attack_idx, :].data.sort(dim=1, descending=True)
                class_score = class_score[:, 1:]
                class_idx = class_idx[:, 1:]

                rand_seed = 1 * (class_score.sum(dim=1) * torch.rand([attack_idx.shape[0]]).cuda()).unsqueeze(1)
                prob = class_score.cumsum(dim=1)
                for k in range(attack_idx.shape[0]):
                    for c in range(prob.shape[1]):
                        if (prob[k, c] >= rand_seed[k]).cpu().numpy():
                            attack_class[k] = class_idx[k, c]
                            break

                # Forward and backward for adversarial samples
                attacked_inputs = Variable(attack.run(t_net, inputs[attack_idx, :, :, :].data, attack_class))
                batch_size2 = attacked_inputs.shape[0]

                attack_out_t = t_net(attacked_inputs)
                attack_out_s = s_net(attacked_inputs)

                
                ## Determine Total variance in the space                 
                # KD loss for Boundary Supporting Samples (BSS)
                if compress:
                    b = attack_out_t.detach().cpu().numpy()
                    ua,sa,vha=np.linalg.svd(b.T,full_matrices=False)
                    loss += - ratio_attack * (F.softmax(torch.tensor(vha.T).cuda() , 1).detach() * F.log_softmax(attack_out_s / temperature, 1)).sum() / batch_size2
                else:
                    loss += - ratio_attack * (F.softmax(attack_out_t / temperature, 1).detach() * F.log_softmax(attack_out_s / temperature, 1)).sum() / batch_size2
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
        _, predicted = torch.max(out_s[0:batch_size1, :].data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().float().sum()
        b_idx = batch_idx
    
    print('1,Train,%d,%.2f,%.3f,%.3f,%s,%d,%d,%s,%s,%d' %  (int(epoch), time.time() - epoch_start_time,train_loss / (b_idx + 1), 100. * correct / total, attack_id, attack_size, temperature,t_net_id,s_net_id, int(compress)))
 

def test(use_cuda, compress, attack_id, temperature, attack_size, t_net_id, s_net_id, testloader, criterion_CE, save_dir, net, epoch, save=False):
    epoch_start_time = time.time()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion_CE(outputs, targets)

        test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().float().sum()
        b_idx= batch_idx

    print('1,Test,%d,%.2f,%.3f,%.3f,%s,%d,%d,%s,%s,%d' %  (int(epoch), time.time() - epoch_start_time,test_loss / (b_idx + 1), 100. * correct / total, attack_id, attack_size, temperature,t_net_id,s_net_id, int(compress)))

    if save:
        # Save checkpoint.
        acc = 100.*correct/total
        if epoch is not 0 and epoch % 80 is 0:
            #print('Saving..')
            state = {
                'net': net if use_cuda else net,
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, './' + save_dir + '/%d_epoch.t7' % epoch)

def main(params): 
    # Parameters
    dataset_name = params.dataset
    t_net_id = params.t_net_id
    s_net_id = params.s_net_id
    save_dir = params.save_dir
    temperature = params.temp
    gpu_num = params.gpu
    attack_size = params.attack_size
    attack_id = params.attack_id
    max_epoch = params.max_epoch
    compress = params.compress
    global optimizer

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    use_cuda = torch.cuda.is_available()
    
    # Dataset
    if dataset_name is params.dataset:
        # CIFAR-10
        #print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
    else:
        raise Exception('Undefined Dataset')

    if t_net_id == 'ResNet26':
        # Teacher network
        teacher = BN_version_fix(torch.load('./results/Res26_C10/320_epoch.t7', map_location=lambda storage, loc: storage.cuda(0))['net'])
        t_net = ResNet26()
        t_net.load_state_dict(teacher.state_dict())
    else :
        print('no teacher net was used')
    
    if s_net_id == 'ResNet8':    
        # Student network
        s_net = ResNet8()
    elif s_net_id == 'ResNet14':    
        # Student network
        s_net = ResNet14()
    elif s_net_id == 'ResNet20':    
        # Student network
        s_net = ResNet20()
    elif s_net_id == 'ResNet26':    
        # Student network
        s_net = ResNet26()
    else :
        print('no student net was used')

    if use_cuda:
        #print('CUDA Available')
        torch.cuda.set_device(gpu_num)
        #torch.cuda.set_device("cuda:" + str(gpu_num))
        t_net.cuda()
        s_net.cuda()
        cudnn.benchmark = True

    if attack_id == 'BSS':   
        # Proposed adversarial attack algorithm (BSS)
        attack = attacks.AttackBSS(targeted=True, num_steps=10, max_epsilon=16, step_alpha=0.3, cuda=True, norm=2)
    else:
        print('no attach was used')

    criterion_MSE = nn.MSELoss(size_average=False)
    criterion_CE = nn.CrossEntropyLoss()
    print('Stage,Method,Epoch,Time,Loss,Acc, Attack, Attack_Size, Temprature, Teacher_Net, Student_Net, Compressed')  # prepare header for csv file results
    for epoch in range(1, max_epoch+1):
        if epoch == 1:
            optimizer = optim.SGD(s_net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        elif epoch == max_epoch/2:
            optimizer = optim.SGD(s_net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        elif epoch == max_epoch/4*3:
            optimizer = optim.SGD(s_net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

        ratio = max(3 * (1 - epoch / max_epoch), 0) + 1
        attack_ratio = max(2 * (1 - 4 / 3 * epoch / max_epoch), 0) + 0

        train_attack_KD(use_cuda, compress, optimizer, attack_id, t_net_id, s_net_id, attack, trainloader, criterion_CE, temperature, attack_size, t_net, s_net, ratio, attack_ratio, epoch)

        test(use_cuda, compress, attack_id, temperature, attack_size, t_net_id, s_net_id, testloader, criterion_CE, save_dir, s_net, epoch, save=True)

    state = {
        'net': s_net,
        'epoch': max_epoch,
    }
    torch.save(state, './' + save_dir + '/%depoch_final.t7' % (max_epoch))


if __name__ == '__main__':
    parser = ArgumentParser(description='Main entry point') 
    parser.add_argument("--dataset", type=str, default='CIFAR-10') 
    parser.add_argument("--t_net_id", type=str, default='ResNet26')
    parser.add_argument("--s_net_id", type=str, default='ResNet20') 
    parser.add_argument("--save_dir", type=str, default='results/BSS_distillation_80epoch_res8_C10')
    parser.add_argument("--gpu", type=int, default=1) 
    parser.add_argument("--temp", type=int, default=3)
    parser.add_argument("--attack_id", type=str, default='BSS')
    parser.add_argument("--attack_size", type=int, default=64)
    parser.add_argument("--max_epoch", type=int, default=10)
    parser.add_argument("--compress", type=int, default=0)
    FLAGS = parser.parse_args()
    np.random.seed(9)
    main(FLAGS)