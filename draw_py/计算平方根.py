import math

def calculate_square_root(number):
    if number < 0:
        return "输入的数字不能为负数"  # 负数没有实数平方根
    return math.sqrt(number)

# 用户输入一个数字
number = float(input("请输入一个数字："))

# 计算并输出平方根
result = calculate_square_root(number)
print(f"{number} 的平方根是 {result}")
