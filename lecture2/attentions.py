import torch.nn as nn
import logging
import math

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_head: int, num_hidden: int, dropout: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_head = num_head
        self.q_proj = nn.Linear(num_hidden, num_hidden, bias=False)  # 单个矩阵乘法不带偏置项
        self.k_proj = nn.Linear(num_hidden, num_hidden, bias=False)
        self.v_proj = nn.Linear(num_hidden, num_hidden, bias=False)
        self.o_proj = nn.Linear(num_hidden, num_hidden, bias=False)
        self.dropout = nn.Dropout(dropout)  # 别忘了dropout
        assert num_hidden % num_head == 0, logger.error(
            f"num_hidden({num_hidden}) cannot be evenly divided by num_head({num_head})")

    @staticmethod
    def attention(Q, K, V, mask=None, dropout=None):
        """
        Q: [bs, num_head, seq_len, head_dim]
        K: [bs, num_key_query_pairs, seq_len, head_dim]
        V: [bs, num_key_query_pairs, seq_len, head_dim]
        """
        head_dim = Q.shape[-1]
        attention_score = Q @ K.transpose(-2, -1) / math.sqrt(head_dim)  # 别忘了除以根号head_dim

        # 别忘了掩码的注意力机制
        if mask is not None:
            attention_score.masked_fill_(mask == 0, -1e-9)

        if dropout is not None:
            attention_score = dropout(attention_score)

        attention_weight = nn.functional.softmax(attention_score, -1)  # [bs, num_key_query_pairs, seq_len, seq_len]
        output = attention_weight @ V  # [bs, num_key_query_pairs, seq_len, head_dim]
        return output

    def forward(self, X, mask=None):
        """
        X: [bs, seq_len, num_hidden]
        """
        num_hidden = X.shape[-1]
        head_dim = num_hidden // self.num_head
        Q = self.q_proj(X).view(X.shape[0], X.shape[1], self.num_head, head_dim).transpose(1, 2)
        K = self.k_proj(X).view(X.shape[0], X.shape[1], self.num_head, head_dim).transpose(1, 2)
        V = self.v_proj(X).view(X.shape[0], X.shape[1], self.num_head, head_dim).transpose(1, 2)

        output = self.attention(Q, K, V, mask, self.dropout)
        concated_output = output.transpose(1, 2).contiguous().view(
            [output.shape[0], output.shape[2], output.shape[1] * output.shape[-1]])
        # 这个地方一定要记得把seq_len还有num_key_query_pairs的位置换回来在做view 不然东西就变了
        # 另外因为output是通过矩阵乘法计算得到的新的张量所以本身他在内存存储上是连续的，
        # 做了transpose之后就不连续了，所以后面要用contiguous()拷贝一份数据创建一个新的在当前stride下连续的张量再做view
        projected_output = self.o_proj(concated_output)
        return projected_output


class GroupQueryAttention(nn.Module):
    def __init__(self, num_head, num_key_value_pairs, num_hidden, dropout: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_head = num_head
        self.num_key_value_pairs = num_key_value_pairs
        assert num_head % num_key_value_pairs == 0, logger.error(
            f"num_head cannot be evenly divided by num_key_value_pair")
        assert num_hidden % num_head == 0, logger.error(f"num_hidden cannot be evenly divided by num_head")
        self.key_value_dim = (num_hidden // num_head) * num_key_value_pairs
        self.q_proj = nn.Linear(num_hidden, num_hidden, bias=False)
        self.k_proj = nn.Linear(num_hidden, self.key_value_dim, bias=False)
        self.v_proj = nn.Linear(num_hidden, self.key_value_dim, bias=False)
        self.o_proj = nn.Linear(num_hidden, num_hidden, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x: [bs, seq_len, num_hidden]
        """
        bs, seq_len, num_hidden = x.shape
        q = self.q_proj(x).view(bs, seq_len, self.num_head, -1).transpose(1, 2)
        k = self.k_proj(x).view(bs, seq_len, self.num_key_value_pairs, -1).transpose(1, 2)
        v = self.v_proj(x).view(bs, seq_len, self.num_key_value_pairs, -1).transpose(1, 2)
        repeat_k = self.repeat_kv(k)
        repeat_v = self.repeat_kv(v)
        o = MultiHeadAttention.attention(q, repeat_k, repeat_v, mask, self.dropout)  # [bs, num_head, seq_len, head_dim]
        o = o.transpose(1, 2).contiguous().view(bs, seq_len, num_hidden)
        output = self.o_proj(o)
        return output

    def repeat_kv(self, x):
        bs, num_key_value_pairs, seq_len, head_dim = x.shape
        time = self.num_head // self.num_key_value_pairs
        if time == 1:
            return x
        x = x.unsqueeze(2).expand(bs, num_key_value_pairs, time, seq_len, head_dim)
        x = x.contiguous().view(bs, num_key_value_pairs * time, seq_len, head_dim)
        return x










