import torch
import torch.nn as nn
import torch.optim as optim
from Data import train_loader,valid_loader,test_loader,loader,rest,max,min,lendict
from Model import Transformer
import time
import os
from tensorboardX import SummaryWriter
import tensorboardX as tb
import numpy
from scipy.stats import pearsonr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max5=max
min5=min
print(max5)
print(min5)

# 定义模型、优化器、损失函数
model = Transformer().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


# 训练
def train():
    model.train()  # Turn on the train mode
    total_loss = 0.
    t_loss = 0.

    start_time = time.time()
    for step, (enc_inputs, dec_inputs, dec_outputs) in enumerate(train_loader):
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, dec_outputs.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        t_loss += loss.item()

        log_interval = 500

        if step % log_interval == 0:
            if( step != 0 ):
                cur_loss = total_loss / log_interval

                elapsed = (time.time() - start_time) / log_interval
            else:
                cur_loss = total_loss

                elapsed = time.time() - start_time

            print('| epoch {:3d} | batches {:5d} | '
                  'lr {:0.5f} | s/batch {:5.2f} | '
                  'loss {:5.2f} '.format(epoch, step, lr_scheduler.get_last_lr()[0], elapsed , cur_loss))
            total_loss = 0
            start_time = time.time()

    return t_loss / len(loader)


def evaluate(eval_model, C_N_CA_angle_valid_loader):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.

    with torch.no_grad():
        for enc_inputs, dec_inputs, dec_outputs in C_N_CA_angle_valid_loader:
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            # print(criterion(outputs, dec_outputs.view(-1)).item())
            total_loss += criterion(outputs, dec_outputs.view(-1)).item()


    return total_loss / len(C_N_CA_angle_valid_loader)


# 贪心策略
def greedy_decoder(model, enc_input, start_symbol):
    """
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    dec_input = dec_input.to(device)
    terminal = False
    next_symbol = start_symbol
    while not terminal:
        dec_input=torch.cat([dec_input.detach().to(device),torch.tensor([[next_symbol]],dtype=enc_input.dtype).to(device)],-1)
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word
        # if next_symbol == tgt_vocab["."]:
        #     terminal = True
        if (next_symbol == 1002):
            terminal = True
        # print(next_word)
    return dec_input


# 测试
# 以下代码要将test_loader的batchsize大小设为1
def write_to_list(b,a,index,c):
    for j in range(c, c + list(lendict.values())[int(index/10)]):
        b[index].append(a[j])
    for j in range(c + list(lendict.values())[int(index/10)], c + 2 * list(lendict.values())[int(index/10)]):
        b[index+1].append(a[j])
    for j in range(c + 2 * list(lendict.values())[int(index/10)], c + 3 * list(lendict.values())[int(index/10)]):
        b[index+2].append(a[j])
    for j in range(c + 3 * list(lendict.values())[int(index/10)], c + 4 * list(lendict.values())[int(index/10)]):
        b[index+3].append(a[j])
    for j in range(c + 4 * list(lendict.values())[int(index/10)], c + 5 * list(lendict.values())[int(index/10)]):
        b[index+4].append(a[j])
    for j in range(c + 5 * list(lendict.values())[int(index/10)], c + 6 * list(lendict.values())[int(index/10)]):
        b[index+5].append(a[j])
    for j in range(c + 6 * list(lendict.values())[int(index/10)], c + 7 * list(lendict.values())[int(index/10)]):
        b[index+6].append(a[j])
    for j in range(c + 7 * list(lendict.values())[int(index/10)], c + 8 * list(lendict.values())[int(index/10)]):
        b[index+7].append(a[j])
    for j in range(c + 8 * list(lendict.values())[int(index/10)], c + 9 * list(lendict.values())[int(index/10)]):
        b[index+8].append(a[j])
    for j in range(c + 9 * list(lendict.values())[int(index/10)], c + 10 * list(lendict.values())[int(index/10)]):
        b[index+9].append(a[j])


def test(data_loader,path):
    model.eval()
    total_acc=0.
    t_acc = 0.
    # 加载模型
    # path_checkpoint = "../model/checkpoint/ckpt_best.pth"  # 最好模型断点路径
    checkpoint = torch.load(path)  # 加载最好模型断点
    model.load_state_dict(checkpoint['model'])
    a=[]
    for i,data in enumerate(data_loader):
        print('i:',i)
        enc_inputs,dec_inputs,dec_outputs=data
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
        greedy_dec_input = greedy_decoder(model, enc_inputs.view(1, -1), start_symbol=1001)
        predict, _, _, _ = model(enc_inputs.view(1, -1), greedy_dec_input)
        predict = predict.data.max(1, keepdim=True)[1]

        predict = predict[0:len(predict)-1]
        for j in range(0,len(predict.squeeze().cpu().numpy().tolist())):

            a.append(predict.squeeze().cpu().numpy().tolist()[j])

        acc=normalized_euclidean(predict.squeeze().cpu().numpy(),dec_outputs[:, 0:50].squeeze().cpu().numpy())
        total_acc += acc
        ac = normalized_euclidean(enc_inputs.squeeze().cpu().numpy(), dec_outputs[:, 0:50].squeeze().cpu().numpy())
        t_acc += ac
        print('source:', enc_inputs.squeeze(), '->', 'prediction:', predict.squeeze(), ' ;  target:', dec_outputs[:, 0:50].squeeze())
        print(acc)
        i += 1

    print(len(a))
    if (rest != None):
        print("rest begin\n")
        print(rest)
        for i in range(0, len(rest)):
            a[len(a) - 1].append(rest[i])

    b = [[] for x in range(400)]
    c = 0
    write_to_list(b, a, 0, c)
    j = 0
    for j in range(0, 39):
        c += 10 * list(lendict.values())[j]
        write_to_list(b, a, (j + 1) * 10, c)

    for i in range(0, len(b)):
        output_file_path = '../result/C_N_CA_angle_pred/'
        output_file_name = str(list(lendict.keys())[int(i / 10)])[0:12] + '_%d' % (i % 10) + '.json'
        with open(output_file_path + output_file_name, "w") as json_output:
            json_output.write(str(b[i]))
        json_output.close()
    # print(len(data_loader))

    print(len(b))
    print(len(b[0]))
    print(len(b[399]))
    print('\nprediction acc:',total_acc/len(data_loader),'\n')
    print('\noriginal acc:', t_acc / len(data_loader), '\n')



# 标准化欧几里得距离——>精度
def normalized_euclidean(p, q):
    same = 0
    for i in range(0, len(p)):
        if i in range(0, len(q)):
            same += 1
    sumnum = 0
    for i in range(same):
        if p[i] == 0 and q[i] == 0:
            sumnum += 0
        else:
            avg = (p[i] - q[i]) / 2
            si = ((p[i] - avg) ** 2 + (q[i] - avg) ** 2) ** 0.5
            sumnum += ((p[i] - q[i]) / si) ** 2
    return 1/(1+sumnum ** 0.5)



# path_checkpoint = "../model/C_N_CA_angle_model/ckpt_best.pth"
# checkpoint = torch.load(path_checkpoint)
# print(checkpoint['epoch'])



if __name__ == "__main__":

    RESUME = False
    EPOCH = 100
    start_epoch = 0
    best_val_loss = float("inf")

    if RESUME:
        path_checkpoint = "../model/C_N_CA_angle_model/ckpt_best.pth"  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        model.load_state_dict(checkpoint['model'])  # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        lr_scheduler.load_state_dict(checkpoint['lr_schedule'])  # 学习率
        loss = checkpoint['loss']  # loss

    # ----------------------------------------tensorboard----------------------------------------
    loss_show = []
    train_writer = SummaryWriter('../result/C_N_CA_angle_plot/loss/train')
    val_writer = SummaryWriter('../result/C_N_CA_angle_plot/loss/val')
    # ----------------------------------------tensorboard----------------------------------------

    for epoch in range(start_epoch + 1, EPOCH+1):
        epoch_start_time = time.time()
        train_loss = train()
        val_loss = evaluate(model, valid_loader)
        # test_loss = evaluate(model, psi_test_loader)

        # ----------------------------------------tensorboard----------------------------------------
        # loss_show.append(val_loss)
        train_writer.add_scalar("loss", train_loss, epoch)
        val_writer.add_scalar("loss", val_loss, epoch)
        # ----------------------------------------tensorboard----------------------------------------

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:7.4f} | valid loss {:7.4f} |  '.format(epoch, (time.time() - epoch_start_time),train_loss,val_loss))
        print('-' * 89)
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "lr_schedule": lr_scheduler.state_dict(),
            "loss": val_loss
        }

        if not os.path.isdir("../model/C_N_CA_angle_model"):
            os.mkdir("../model/C_N_CA_angle_model")
        # torch.save(checkpoint, '../model/C_N_CA_angle_model/ckpt_%s.pth' % (str(epoch)))
        lr_scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            torch.save(checkpoint, '../model/C_N_CA_angle_model/ckpt_best.pth')

        # print('----------CURRENT CKPT PREDICTION RESULT-----------')
        # current_epoch_checp = '../model/checkpoint/ckpt_%s.pth' % (str(epoch))
        # test(psi_test_loader, current_epoch_checp)
        # print('----------------------------------------------------')
    '''

    print('----------PREDICTION-----------')
    test(test_loader,'../model/C_N_CA_angle_model/ckpt_best.pth')
    # test(test_loader1,'../model/C_N_CA_angle_model/ckpt_best.pth')

'''


