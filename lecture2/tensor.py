import  torch
from einops import repeat, rearrange, einsum

num_gpus = torch.cuda.device_count()  # @inspect num_gpus
for i in range(num_gpus):
    properties = torch.cuda.get_device_properties(i)
    print(i, properties)

# # 变量新建
# x = torch.randn(size=[2,3,4])
# print("x: ", x)
# print("x's shape:", x.shape)
# print("x is contiuous: ", x.is_contiguous())
# print("flattern x:", x.view(-1))
# print("x's stride", x.stride()) # (12, 4, 1)分别对应了x的3个维度在按对应维度的索引访问的时候的步伐

# # 步伐
# dim1, dim2, dim3 = 0, 1, 2
# index = dim1 * x.stride(0) + dim2*x.stride(1) + dim3*x.stride(2)  # 0*12 + 1*4 + 2*1
# assert index == 6
# assert x[dim1,dim2,dim3] == x.view(-1)[index]
# print(x[dim1,dim2,dim3])

# 内存共享与拷贝
def same_storge(x:torch.Tensor, y:torch.Tensor, x_name:str, y_name:str):
    is_same_storage = x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()
    print(f"Are {x_name} and {y_name} saved together:", is_same_storage)
    return is_same_storage


# y = x.view(4,3,2) 
# same_storge(x, y, "x", "y")
# print("y", y)
# print("shape of y", y.shape)
# print("y's stride", y.stride()) # view 抛弃了原来的stride，直接用新设置的步伐在连续的存储的数组里取数构建矩阵



# x_t = x.transpose(0,2) # 4, 3, 2
# same_storge(x, x_t,  "x", "x_t")
# print("x_t", x_t)
# print("shape of x_t", x_t.shape)
# print(x_t.stride()) # transpose() 保留了原来的stride的组合，通过更换stride的维度，来实现矩阵的重构，所以构建出来的矩阵的在同一维度的邻居，在原来的维度里是同一步伐下的对应的一组值

# # reshape
# x_reshape = x.reshape(4,3,2)
# same_storge(x, x_reshape, "x", "x_reshape")
# print("x_reshape:", x_reshape)
# print("shape of x_reshape:", x_reshape.shape)
# print(x_reshape.stride()) # reshape在操作上几乎和view是一致的，唯一的区别就是如果原来的张量不连续的话，reshape会新拷贝一份数据，返回一个有新指定形状的张量
# print("is y equal to x_reshape:", torch.equal(y, x_reshape))


# x_permute = x.permute(2,1,0)
# same_storge(x, x_permute, "x", "x_permute")
# print("x_permute:", x_permute)
# print("shape of x_permute:", x_permute.shape)
# print(x_permute.stride()) # permute 和 transpose 操作逻辑类似，但是可以指定任一维度的交换
# print("is x_t equal to x_permute:", torch.equal(x_t, x_permute))
# print("is x_permute contiguous:", x_permute.is_contiguous())


# z = x.transpose(0,2).transpose(0,2) # 用transpose交换两次，把维度换回来
# print("is x equal to z:", torch.equal(x, z))
# print("z:",z)
# print("is z contiguous:", z.is_contiguous())



# ================ 测试先view增加维度transpose后再transpose回来view合并回维度的效果 =========================

# x = torch.randn(size=[2,3,8])
# print("x: ", x)
# print("x's shape:", x.shape)
# print("x is contiuous: ", x.is_contiguous())
# print("flattern x:", x.view(-1))
# print("x's stride", x.stride()) 

# k = x.view(2,3,2,4)
# print("k:", k)
# print("k's shape:", k.shape)
# print("is k contiguous:", k.is_contiguous())
# print("stride of k", k.stride())

# k_t = k.transpose(1, 2)
# print("k_t:", k_t)
# print("k_t's shape:", k_t.shape)
# print("is k_t contiguous:", k_t.is_contiguous())
# print("stride of k_t", k_t.stride())

# k_retrans = k_t.transpose(1,2)
# print("k_retrans:", k_retrans)
# print("k_retrans's shape:", k_retrans.shape)
# print("is k_retrans contiguous:", k_retrans.is_contiguous())
# print("stride of k_retrans", k_retrans.stride())
# print("is k_retrans equal to k:", torch.equal(k_retrans, k))
# same_storge(k_retrans,k, "k_retrans", "k")

# x_recovery = k_retrans.view(2,3,8)
# print("x_recovery:", x_recovery)
# print("x_recovery's shape:", x_recovery.shape)
# print("is x_recovery contiguous:", x_recovery.is_contiguous())
# print("stride of x_recovery", x_recovery.stride())
# print("is x_recovery equal to x:", torch.equal(x_recovery, x))
# same_storge(x_recovery,x, "x_recovery", "x")



# # ================ 测试矩阵乘法是否会影响张量存储的连续性 =========================
# x = torch.randn(size=[4,3,10])
# q = x.view(4,3,2,5).transpose(1,2) # [4,2,3,5]
# k = x.view(4,3,2,5).transpose(1,2) # [4,2,3,5]
# v = x.view(4,3,2,5).transpose(1,2) # [4,2,3,5]
# print(q.is_contiguous(), k.is_contiguous())
# a = q @ k.transpose(-2,-1) # [4,2,3,3]
# print(a.is_contiguous())
# o = a @ v
# print(o.is_contiguous())
# same_storge(o,a,"o","a")
# same_storge(o,a,"o","v")
# # 结论 矩阵乘法新建了一个存储对象


# ================ 查看contiguous的拷贝和新建的过程 ============================
# x = torch.randn(size=[2,3,4])
# print(x.shape)
# print(x.stride())
# print(x.view(-1))
# x_t = x.transpose(1,2) # [2,4,3]
# print(x_t.stride()) # [12,1,4]
# x_t_contiguous = x_t.contiguous()
# print(x_t_contiguous.shape)
# print(x_t_contiguous.stride()) # [12,1,4]
# print(x_t_contiguous.view(-1))

# print(x)
# print(x_t_contiguous)
# print(x_t_contiguous.view(2,2,2,3))


# # =============== 查看unsqueeze/squeeze/expand/repeat对张量的处理过程
# print("x:")
# x = torch.randn(size=[2,3,4])
# print(x.shape)
# print(x.stride())
# print(x.view(-1))

# print("="*50)
# print("unsqueezed_x:")
# unsqueezed_x = x.unsqueeze(1)
# print("shape:", unsqueezed_x.shape)
# same_storge(x, unsqueezed_x, "x", "unsqueezed_x")
# print(unsqueezed_x.stride())
# print(unsqueezed_x.is_contiguous())
# print(unsqueezed_x)

# print("="*50)
# print("expanded_x:")
# expanded_x = unsqueezed_x.expand(size=[2,2,3,4]) # 这里可以尝试切换成expanded_x = x.expand(size=[2,2,3,4])对比一下效果看看
# same_storge(x, expanded_x, "x", "expanded_x")
# print(expanded_x.stride())
# print(expanded_x.is_contiguous())
# print(expanded_x)
# # 直接在x上expand 1 2  以后会变成 1 2 1 2
# # 先unsqueeze再expand就会变成 1 1 2 2
# # 另外 expand 操作会导致行索引在张量的存储对象上不连续,但还是在原来的存储对象上做处理


# print("="*50)
# print("repeat_x:")
# repeat_x = unsqueezed_x.repeat(1,2,1,1) 
# same_storge(x, repeat_x, "x", "repeat_x")
# print(repeat_x.shape)
# print(repeat_x.stride())
# print(repeat_x.is_contiguous())
# print(repeat_x)
# print(torch.equal(repeat_x, expanded_x))
# # unsqueeze在做expand或repeat拿到的矩阵几乎是一样的
# # 除了expand和repeat在函数输入上不一样以外，repeat操作会新建一个存储的张量，但是expand会在原来的对象上做视图的变更，且expand是不连续的，但是repeat是连续的

# print("="*50)
# print("squeezed_x:")
# squeezed_x = repeat_x.view(4,3,4)
# same_storge(squeezed_x, repeat_x, "x", "squeezed_x")
# print(squeezed_x.shape)
# print(squeezed_x.stride())
# print(squeezed_x.is_contiguous())
# print(squeezed_x)

# # ========================================== 对比flatten和view(-1)的区别========================
# x = torch.randn(size=[2,3,4])
# y = x.view(-1)
# z = x.flatten()

# same_storge(x,y,"x","y")
# same_storge(x,z,"x","z")
# print(torch.equal(y,z)) # 结论：矩阵一样的，且都创建的是视图而不是一个新的张量。
# torch.index_select()


# =========================================== 对比张量先广播后transpose 与直接expand好对应维度之间的区别
inv_freq =  torch.tensor([1 / (10000 ** (2 * k / 10)) for k in range(10 // 2 )])
token_positions = torch.arange(6).unsqueeze(0)
print(inv_freq)
print(token_positions)
print("="*50)

inv_freq_expanded = repeat(inv_freq.float(), "half_dk -> bs half_dk 1", bs=token_positions.shape[0]) # 重复频率,每个seq一份inv_freq
position_ids_expanded = rearrange(token_positions.float(), "bs seq_len -> bs 1 seq_len") # 给position_ids拓展维度，方便做广播
print(inv_freq_expanded)
print(position_ids_expanded)
freqs = einsum(inv_freq_expanded, position_ids_expanded,
        "bs half_dk x, bs x seq_len -> bs half_dk seq_len"
    ).transpose(1,2)
print(freqs)
print("="*50)

inv_freq_expanded_v1 = repeat(inv_freq.float(), "half_dk -> bs 1 half_dk", bs=token_positions.shape[0]) # 重复频率,每个seq一份inv_freq
position_ids_expanded_v1 = rearrange(token_positions.float(), "bs seq_len -> bs seq_len 1") # 给position_ids拓展维度，方便做广播
print(inv_freq_expanded_v1)
print(position_ids_expanded_v1)
freqs_v1 = einsum(position_ids_expanded_v1, inv_freq_expanded_v1,
        "bs seq_len x, bs x half_dk -> bs seq_len half_dk"
    )
print(freqs_v1)
print("="*50)
print("is equal:", freqs.equal(freqs_v1))
