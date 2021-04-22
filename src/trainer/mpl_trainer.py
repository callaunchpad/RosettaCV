import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Meta psuedo labels training loop

weight_u: multiplier on labeled data for teacher

NOTE: Expects dl to have batch size of 1 for unlabeled and labeled data
'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_mpl(teacher_model, student_model, unlabeled_dl, labeled_dl, dataset, num_epochs=5, learning_rate=1e-3, weight_u=1, save_model=False):
    # Setup definitions
    t_optimizer = torch.optim.Adam(teacher_model.parameters(), lr=learning_rate, weight_decay=1e-5)
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
            labeled_iter = iter(labeled_dl)
            unlabeled_iter = iter(unlabeled_dl)

            for i in range(num_iter):
                # We get one unlabeled image and one labeled image
                image_l, label = next(labeled_iter)
                image_u, _ = next(unlabeled_iter)

                # 0) resize image to what PyTorch wants and conver to device
                if dataset == 'fashion_mnist':
                    image_l = image_l.view(-1, 1, 28, 28).to(device)
                    image_u = image_u.view(-1, 1, 28, 28).to(device)
                elif dataset == 'imagenet':
                    image_l = image_l.view(-1, 1, 224, 224).to(device)
                    image_u = image_u.view(-1, 1, 224, 224).to(device)
                
                label = label.type(torch.LongTensor).to(device)

                # 1) pass labeled image through teacher and save the loss for future backprop
                t_logits = teacher_model(image_l)
                t_l_loss = F.cross_entropy(t_logits, label)

                # 2) pass labeled image through student and save the loss
                with torch.no_grad(): # we don't want to update student
                    s_logits_l = student_model(image_l)
                s_l_loss_1 = F.cross_entropy(s_logits_l, label)

                # 3) generate pseudo labels from teacher
                mpl_image_u = teacher_model(image_u)

                # 4) pass unlabeled through student, calculate gradients, and optimize student
                s_logits_u = student_model(image_u)
                s_mpl_loss = F.binary_cross_entropy_with_logits(s_logits_u, mpl_image_u)
                s_mpl_loss.backward() # calculate gradients
                s_optimizer.step() # step in the direction of gradients
                # We will clear out gradients at the end

                # 5) pass labeled data through updated student and save the loss
                with torch.no_grad(): # we don't want to update student
                    s_logits_l_updated = student_model(image_l)
                s_l_loss_2 = F.cross_entropy(s_logits_l_updated, label)
                t_mpl_loss = s_l_loss_2 - s_l_loss_1 # NOTE: Not sure about this, maybe it is other direction?

                # 6) calculate teacher loss and optimizer teacher
                t_loss = (weight_u * t_l_loss) + t_mpl_loss
                t_loss.backward()
                t_optimizer.step()
                # We will clear out gradients at the end

                # 7) clear out gradients of teacher and student
                teacher_model.zero_grad()
                student_model.zero_grad()

                # 8) display current training information
                global_step += 1
                batch_num += 1
                if global_step % 100 == 0:
                    print('Epoch:{} Batch:{}/{} Teacher Loss:{:.4f} Student Loss:{:.4f}'.format(epoch+1, batch_num, total_batches, float(t_loss)/batch_num, float(s_l_loss_1)/batch_num))
                    #wandb.log({ 'batch_loss': float(loss) })
            
            # display information for each epoch
            print('Epoch:{} Teacher Loss:{:.4f} Student Loss:{:.4f}'.format(epoch+1, float(t_loss)/batch_num, float(s_l_loss_1)/batch_num))

    else:
        print('[!] More labeled data than unlabeled')