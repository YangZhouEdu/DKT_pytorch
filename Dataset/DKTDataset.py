import numpy as np
from torch.utils.data import Dataset

class DKTDataset(Dataset):
    def __init__(self,group,n_skill,max_seq = 100,min_step=10):
        super(DKTDataset, self).__init__()
        self.max_seq = max_seq
        self.n_skill = n_skill
        self.samples = group
        self.min_step = min_step
        self.user_ids = []
        for user_id in group.index:
            q, qa = group.loc[user_id]

            if len(q) < 10:
                continue
            self.user_ids.append(user_id)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        q_, qa_ = self.samples.loc[user_id]
        seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        if seq_len >= self.max_seq:
            q[:] = q_[-self.max_seq:]
            qa[:] = qa_[-self.max_seq:]
        else:
            q[-seq_len:] = q_
            qa[-seq_len:] = qa_

        target_id = q[1:]
        label = qa[1:]  # 预测值

        # x = np.zeros(self.max_seq - 1, dtype=int)
        x = q[:-1].copy()
        answers = qa[:-1].copy()
        # x += (qa[:-1] == 1) * self.n_skill # 特征值
        onehot = self.onehot(x, answers)


        return  onehot, target_id, label

    def onehot(self, questions, answers):
        result = np.zeros(shape=[self.max_seq -1  , 2 * self.n_skill])
        for i in range(self.max_seq - 1):
            if answers[i] > 0:
                result[i][questions[i]] = 1
            elif answers[i] == 0:
                result[i][questions[i] + self.n_skill] = 1
        return result
