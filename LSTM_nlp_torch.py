import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

# torchtext
from torchtext import data
from torchtext import vocab
from torchtext import datasets

import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

################################################################
# データセット
################################################################

# torchtextでは、データを取り込む際、
# ・各データソースの各カラムに対して前処理を管理するFieldクラスを指定
# ・各カラムごとに、指定されたFieldクラスが管理する前処理が実行される
# Fieldクラスで、読み込んだデータに施す前処理とその結果を管理

# sequential: 対応するデータがtextのように可変長のデータかどうか
# lower: 文字を小文字に変換するかどうか
# tokenize: トークン化を指定する関数
# include_lengths:
# batch_first: (batch, )
# fix_length: 1文の長さ、もし短い文がある場合は、<pad>でパディングされる
TEXT = data.Field(sequential=True, tokenize=(lambda s: s.split()), lower=True, include_lengths=True, batch_first=True, fix_length=200)
LABEL = data.LabelField()

train_dataset, test_dataset = datasets.IMDB.splits(TEXT, LABEL, root='data')
train_dataset, val_dataset = train_dataset.split()



################################################################
# 単語の数値化、embedding
################################################################

# build_vocabで、単語の辞書化
# 学習済みembeddingを利用、Wikipediaで学習された300次元のGloveのベクトル
# min_freq: 出現頻度が低い単語は無視
TEXT.build_vocab(train_dataset, min_freq=3, vectors=vocab.GloVe(name='6B', dim=300))
LABEL.build_vocab(train_dataset)


# データをバッチ化
X_train, X_val, X_test = data.BucketIterator.splits((train_dataset, val_dataset, test_dataset), batch_size=32, sort_key=(lambda x: len(x.text)), repeat=False, shuffle=True)



# モデル
class LSTMClassifier(nn.Module):
    def __init__(self, batch_size, hidden_size, output_size, vocab_size, embedding_length, weights):
        super(LSTMClassifier, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_length)
        self.embedding.weight.data.copy_(weights)  # embeddingの初期化

        self.lstm = nn.LSTM(embedding_length, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 単語を数値化
        x = self.embedding(x)

        # 初期隠れ状態、セル状態を設定
        h0 = torch.zeros(1, self.batch_size, self.hidden_size)
        c0 = torch.zeros(1, self.batch_size, self.hidden_size)

        output_seq, (h_n, c_n) = self.lstm(x, (h0, c0))
        # output_seq: shape=(batch, seq_length, output_dim)

        # 最後のタイムステップの隠れ状態をデコード
        out = self.fc(h_n[-1])

        return out


batch_size = 32
hidden_size = 256
output_size = 2  # positive, negativeのラベル
# 単語数
vocab_size = len(TEXT.vocab)
# embedding
embedding_length = 300
word_embedding = TEXT.vocab.vectors


# model
model = LSTMClassifier(batch_size, hidden_size, output_size, vocab_size, embedding_length, word_embedding)
model.to(device)

# 損失関数、最適化手法
criterion = nn.CrossEntropyLoss()

learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



def train(train_loader):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for i, batch in enumerate(train_loader):
        text = batch.text[0]
        text.to(device)

        if (text.size()[0] != 32):
            continue

        labels = batch.label
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(text)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        correct += (outputs.max(1)[1]==labels).sum().item()
        loss.backward()
        optimizer.step()
        total += labels.size(0)

    train_loss = running_loss / len(X_train)
    train_acc = correct / total *100

    return train_loss, train_acc


def valid(test_loader):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            text = batch.text[0]
            text.to(device)

            if (text.size()[0] != 32):
                continue

            labels = batch.label
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(text)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            correct += (outputs.max(1)[1]==labels).sum().item()
            loss.backward()
            optimizer.step()
            total += labels.size(0)

        val_loss = running_loss / len(X_val)
        val_acc = correct / total * 100

    return val_loss, val_acc



num_epochs = 10
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

for epoch in range(num_epochs):
    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0

    train_loss, train_acc = train(X_train)
    val_loss, val_acc = train(X_val)

    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)


model.eval()
with torch.no_grad():
    total = 0
    test_acc = 0
    for batch in X_test:
        text = batch.text[0]
        text.to(device)
        if (text.size()[0] != 32):
            continue
        labels = batch.label
        labels = labels.to(device)

        outputs = model(text)
        test_acc += (outputs.max(1)[1] == labels).sum().item()
        total += labels.size(0)

    print('Test ACC: %.3f'%(test_acc/total*100))




# plot learning curve
plt.figure()
plt.plot(range(num_epochs), loss_list, 'r-', label='train_loss')
plt.plot(range(num_epochs), val_loss_list, 'b-', label='val_loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()


plt.figure()
plt.plot(range(num_epochs), train_acc_list, 'r-', label='train_acc')
plt.plot(range(num_epochs), val_acc_list, 'b-', label='val_acc')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.grid()
plt.show()
