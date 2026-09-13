from torch import nn
import torch
class feedfowardlayer(nn.Module):
    def __init__(self,hidden_size,ff_size,dropout):
        super().__init__()

        self.ff_layer=nn.Sequential(
            nn.Linear(hidden_size,ff_size),

            nn.ReLU(),

            nn.Linear(ff_size,hidden_size)
        )
    def forward(self,attention_out):
        feed_out=self.ff_layer(attention_out)
        return feed_out
        pass
    pass
class MultiHeadAttention(nn.Module):
    def __init__(self, token_dim: int, num_heads: int, dropout: float, device):
        super().__init__()
        """
        多头注意力层
        前馈网络层
        """
        self.token_dim=token_dim#query的嵌入维度
        self.num_heads=num_heads#注意力头的数量
        self.device=device

        self.q=nn.Linear(token_dim, token_dim)
        self.k=nn.Linear(token_dim, token_dim)
        self.v=nn.Linear(token_dim, token_dim)

        self.head_dim= token_dim // num_heads

        self.head_linear=nn.Linear(token_dim, token_dim)

        self.dropout=nn.Dropout(dropout)
        pass
    def scaled_dot_product_attn(self,q,k,v,mask):
        """
        该函数接收的q，k,v都是经过线性变换后的
        """
        # 先说去缩放尺寸
        sqrt_dim= self.token_dim ** 0.5

        # 矩阵乘法过后tmp.shape=[batch_size,num_heads,len_q,len_q]
        tmp=torch.matmul(q,k.permute(0,1,3,2))/sqrt_dim

        if mask is not None:
            tmp=tmp.masked_fill(mask==0,-1e10)
            pass

        softmax_out=torch.softmax(tmp,dim=-1)


        # 矩阵乘法过后=[batch_size,num_heads,len_q,head_dim]
        attention=torch.matmul(self.dropout(softmax_out),v)

        # 开始恢复attention的形状
        # [batch_size,num_heads,len_q,head_dim]->[batch_size,len_q,num_heads,head_dim]->[batch_size,len_q,token_dim]
        attention=attention.view(0,2,1,3).contiguous().view(attention.shape[0], -1, self.token_dim)#调整数值内存位置

        return attention
        pass
    def forward(self,query,key,value,mask=None):
        # query.shape=[batch_size,len_q,token_dim]
        batch=query.shape[0]
        q=self.q(query)#.shape=[batch_size,len_q,token_dim]
        k=self.k(key)
        v=self.v(value)


        # 进行形状调整维缩放点积注意力做形状准备
        # [batch_size, len_q, token_dim]
        # ->[batch_size,(len_q*token_dim/num_heads/token_dim*num_head)=len_q,num_heads,(token_dim/num_heads)=head_dim]
        # =[batch_size,len_q,num_heads,head_dim]
        # ->[batch_size,num_heads,len_q,head_dim]
        q=q.view(batch,-1,self.num_heads,self.head_dim).permute(0,2,1,3)
        k=k.view(batch,-1,self.num_heads,self.head_dim).permute(0,2,1,3)
        v=v.view(batch,-1,self.num_heads,self.head_dim).permute(0,2,1,3)

        attention=self.scaled_dot_product_attn(q,k,v,mask)

        attention=self.head_linear(attention)

        return attention
        pass
    pass



class transformer_encoder_layer(nn.Module):
    def __init__(self,model_dim,ten_num_heads,dropout,ff_size):
        super().__init__()
        """
        该来包含注意力编码层和前向传播编码层
        """
        self.attention_encoder(ten_model_dim,)
        pass
    pass
class transformer_encoder(nn.Module):
    """
    该类包含原始token输入后要先embedding->post位置编码->多头注意力机制->前馈网络
    """
    def __init__(self,
                    src_vocab_size,#嵌入矩阵大小
                    dim_model,#每个toekn的维度，
                    num_heads:int=8,#注意力头
                    feedforward_dim:int=2048,#前馈网络维度
                    dropout: float =0.1
                    ):
        super().__init__()
        pass

    pass