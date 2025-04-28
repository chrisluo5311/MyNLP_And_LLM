import torch

if __name__ == '__main__':
    print(torch.randperm(10))
    a = torch.tensor([True, False, True])

    print(a.int())  # tensor([1, 0, 1], dtype=torch.int32)
    b = a.long()
    print(b)  # tensor([1, 0, 1], dtype=torch.int64)
    # print data type
    print(b.dtype)  # torch.int64

