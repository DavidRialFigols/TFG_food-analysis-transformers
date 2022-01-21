class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return 

    def compute_svd():
        w_q = self.qkv.weight[:768]
        w_k = self.qkv.weight[768:768*2]
        w_v = self.qkv.weight[768*2:768*3]
        try: 
            Uq, Sq, Vhq = torch.linalg.svd(w_q) # SVD decomposition of wq matrix
            Uk, Sk, Vhk = torch.linalg.svd(w_k) # SVD decomposition of wk matrix
            Uv, Sv, Vhv = torch.linalg.svd(w_v) # SVD decomposition of wv matrix
        except:
            print(w_q)
            Uq, Sq, Vhq = torch.linalg.svd(w_q) # SVD decomposition of wq matrix
            Uk, Sk, Vhk = torch.linalg.svd(w_k) # SVD decomposition of wk matrix
            Uv, Sv, Vhv = torch.linalg.svd(w_v) # SVD decomposition of wv matrix
        for i in range(len(Sq)):
            if Sq[i][i] < 0.5:
                Sq[i][i] = 0
            if Sk[i][i] < 0.5:
                Sk[i][i] = 0
            if Sv[i][i] < 0.5:
                Sv[i][i] = 0
        self.qkv.weight[:768] = Uq @ Sq @ Vhq # reconstruct SVD decomposition of wq matrix
        self.qkv.weight[768:768*2] = Uk @ Sk @ Vhk # reconstruct SVD decomposition of wk matrix
        self.qkv.weight[768*2:768*3] = Uv @ Sv @ Vhv # reconstruct SVD decomposition of wv matrix