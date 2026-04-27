import torch

# Проверка наличия доступных GPU
print(f"GPU доступен: {torch.cuda.is_available()}")

# Количество доступных GPU
print(f"Количество доступных GPU: {torch.cuda.device_count()}")

# Имя текущего GPU
if torch.cuda.is_available():
    print(f"Текущий GPU: {torch.cuda.get_device_name(0)}")

# Индекс текущего активного устройства
print(f"Индекс текущего GPU: {torch.cuda.current_device()}")


x = torch.rand(10)
print(x.device) # Выводит 'cpu'
x = x.cuda()
print(x.device) # Должно выводить 'cuda:0'


