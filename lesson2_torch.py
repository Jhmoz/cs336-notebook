import  torch

num_gpus = torch.cuda.device_count()  # @inspect num_gpus
for i in range(num_gpus):
    properties = torch.cuda.get_device_properties(i)
    print(i, properties)

# 变量新建
x = torch.randn(size=[2,3,4])
print("x: ", x)
print("x's shape:", x.shape)
print("x is contiuous: ", x.is_contiguous())
print("flattern x:", x.view(-1))
print("x's stride", x.stride()) # (12, 4, 1)分别对应了x的3个维度在按对应维度的索引访问的时候的步伐

# 步伐
dim1, dim2, dim3 = 0, 1, 2
index = dim1 * x.stride(0) + dim2*x.stride(1) + dim3*x.stride(2)  # 0*12 + 1*4 + 2*1
assert index == 6
assert x[dim1,dim2,dim3] == x.view(-1)[index]
print(x[dim1,dim2,dim3])

# 内存共享与拷贝
def same_storge(x:torch.Tensor, y:torch.Tensor, x_name:str, y_name:str):
    is_same_storage = x.untyped_storage() .data_ptr() == y.untyped_storage() .data_ptr()
    print(f"Are {x_name} and {y_name} saved together:", is_same_storage)
    return is_same_storage


y = x.view(4,3,2)
same_storge(x, y, "x", "y")
print("y", y)
print("shape of y", y.shape)
print("y's stride", y.stride()) # view 抛弃了原来的stride，直接新设置的步伐在连续的存储的数组里取数构建矩阵



x_t = x.transpose(0,2) # 4, 3, 2
same_storge(x, x_t,  "x", "x_t")
print("x_t", x_t)
print("shape of x_t", x_t.shape)
print(x_t.stride()) # transpose() 保留了原来的stride的组合，通过更换stride的维度，来实现矩阵的重构，所以构建出来的矩阵的在同一维度的邻居，在原来的维度里是同一步伐下的对应的一组值