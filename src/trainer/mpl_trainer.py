import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from utils import augmentations
import os
import pdb


import wandb

'''
Meta psuedo labels training loop

weight_u: multiplier on pseudolabel impact

NOTE: Expects dl to have batch size of 1 for unlabeled and labeled data
'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_mpl(teacher_model,
              student_model,
              labeled_dl,
              unlabeled_dl,
              batch_size,
              dataset,
              num_epochs=10,
              learning_rate=1e-3,
              uda_threshold=1.0,
              weight_u=1,
              n_student_steps=1,
              t_optimizer=None,
              s_optimizer=None,
              version='v1',
              model_save_dir=None):
    # Setup wandb
    config = wandb.config
    config.update({
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "uda_threshold": uda_threshold,
        "weight_u": weight_u,
        "n_student_steps": n_student_steps,
    })

    # Setup definitions
    if not t_optimizer:
        t_optimizer = torch.optim.Adam(teacher_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    if not s_optimizer:
        s_optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate, weight_decay=1e-5)

    teacher_model.train()
    student_model.train()

    num_labeled = len(labeled_dl)
    num_unlabeled = len(unlabeled_dl)

    if num_labeled <= num_unlabeled:
        num_iter = num_labeled

        # Iterate for num_epochs
        global_step = 0
        total_batches = num_iter
        for epoch in range(num_epochs):
            batch_num = 0
            running_teacher_loss = 0.0
            running_student_loss = 0.0
            labeled_iter = iter(labeled_dl)
            unlabeled_iter = iter(unlabeled_dl)
            
            # pdb.set_trace()

            for i in range(num_iter):
                # We get one unlabeled image and one labeled image batch
                image_l, label = next(labeled_iter)
                images_u, _ = next(unlabeled_iter)

                image_l = image_l / 255
                # images_u = image_u / 255
                image_u, image_u_aug = images_u

                # 0) resize image to what PyTorch wants and convert to device
                if dataset == 'fashion_mnist':
                    image_l = image_l.view(-1, 1, 28, 28).to(device)
                    image_u = image_u.view(-1, 1, 28, 28).to(device)
                elif dataset == 'imagenet':
                    image_l = image_l.view(-1, 3, 224, 224).to(device)
                    image_u = image_u.view(-1, 3, 224, 224).to(device)
                elif dataset == 'cifar10':
                    image_l = image_l.view(-1, 3, 32, 32).to(device)
                    image_u = image_u.view(-1, 3, 32, 32).to(device)

                label = label.type(torch.LongTensor).to(device)

                # 1) pass labeled image through teacher and save the loss for future backprop
                t_logits = teacher_model(image_l)
                t_l_loss = F.cross_entropy(t_logits, label)

                # 2) pass labeled image through student and save the loss
                with torch.no_grad(): # we don't want to update student
                    s_logits_l = student_model(image_l)
                s_l_loss_1 = F.cross_entropy(s_logits_l, label) # cross_entropy interally takes the average

                # 3) generate pseudo labels from teacher
                for _ in range(n_student_steps):
                    mpl_image_u = teacher_model(image_u)
                    soft_mpl_image_u = torch.softmax(mpl_image_u.detach(), dim=-1) # don't propagate gradients into teacher so use .detach()

                    # 4) pass unlabeled through student, calculate gradients, and optimize student
                    s_logits_u = student_model(image_u)
                    s_mpl_loss = F.binary_cross_entropy_with_logits(s_logits_u, soft_mpl_image_u.detach())
                    s_mpl_loss.backward() # calculate gradients for student network
                    s_optimizer.step() # step in the direction of gradients for student network
                    s_optimizer.zero_grad()
                # We will clear out gradients at the end

                # 5) pass labeled data through updated student and save the loss
                with torch.no_grad(): # we don't want to update student
                    s_logits_l_updated = student_model(image_l)
                s_l_loss_2 = F.cross_entropy(s_logits_l_updated, label) # cross_entropy interally takes the average

                # more details about the mpl loss: https://github.com/google-research/google-research/issues/534
                # NOTE: I'm using soft labels (requires BCE not just CE) instead of hard labels to train so there may be some differences with the reference code
                # the difference between the losses is an approximation of the dot product via a taylor expansion
                dot_product = s_l_loss_2 - s_l_loss_1
                # with hard labels, use log softmax trick from REINFORCE to compute gradients which we then scale with the dot product
                # http://stillbreeze.github.io/REINFORCE-vs-Reparameterization-trick/
                # with soft labels, I have no idea how we can propogate the gradients into the teacher so I use hard labels here
                max_probs, hard_pseudo_label = torch.max(mpl_image_u.detach(), dim=-1)
                t_mpl_loss = dot_product * F.cross_entropy(mpl_image_u, hard_pseudo_label)

                # 6) calculate unsupervised distribution alignment (UDA) loss of teacher on unlabeled images
                t_logits_image_u_aug = teacher_model(image_u_aug)
                uda_loss_mask = (max_probs >= uda_threshold).float()
                t_uda_loss = torch.mean(
                    -(soft_mpl_image_u*torch.log_softmax(t_logits_image_u_aug, dim=-1)).sum(dim=-1)*uda_loss_mask
                )

                # 6) calculate teacher loss and optimizer teacher
                t_u_loss = t_mpl_loss + t_uda_loss
                t_loss = t_l_loss + (weight_u * t_u_loss)
                t_loss.backward()
                # DEBUG: print(teacher_model.encoder.gate[0].weight.grad) # verify that the teacher has a gradient
                t_optimizer.step()
                # We will clear out gradients at the end

                # 7) clear out gradients of teacher and student
                teacher_model.zero_grad()
                student_model.zero_grad()

                # 8) display current training information and update batch information
                global_step += 1
                batch_num += 1
                running_teacher_loss += t_loss.item()
                running_student_loss += s_l_loss_2.item()
                if global_step % 100 == 0:
                    print('Epoch:{} Batch:{}/{} Teacher Loss:{:.4f} Student Loss:{:.4f}'.format(epoch+1, batch_num, total_batches, 
                        t_loss.item(), s_l_loss_2.item()))
                    wandb.log({ 'batch_teacher_loss': t_loss.item(), 'batch_student_loss': s_l_loss_2.item() })
            
            # display information for each epoch
            print('Epoch:{} Teacher Loss:{:.4f} Student Loss:{:.4f}'.format(epoch+1, running_teacher_loss / num_iter, running_student_loss / num_iter))
            wandb.log({ 'epoch': epoch + 1, 'teacher_loss': running_teacher_loss / num_iter, 'student_loss': running_student_loss / num_iter })

        if model_save_dir is not None:
            checkpoint = {
                't_optimizer': t_optimizer.state_dict(),
                's_optimizer': s_optimizer.state_dict(),
                'teacher_model': teacher_model.state_dict(),
                'student_model': student_model.state_dict()
            }
            filename =f"{version}-checkpoint-{str(num_epochs)}-{datetime.now().strftime('%m-%d')}.pt"
            os.makedirs(model_save_dir, exist_ok=True)
            # torch.save(checkpoint, 'trained_models/' + dataset + '/mpl/' + version + '-checkpoint-' + str(num_epochs) + '-' + datetime.now().strftime('%m-%d') + '.pt')
            torch.save(checkpoint, os.path.join(model_save_dir, filename))
    else:
        print('[!] More labeled data than unlabeled')
