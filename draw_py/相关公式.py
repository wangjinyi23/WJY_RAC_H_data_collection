import numpy as np

# 计算反正弦值
result = np.arcsin()

a=9.61
b=0.16
d_m_k=50
h=50

#任意两点之间的Los概率
PLos_m_k = 1 / ( 1 + a * np.exp(-b *  [180 / np.pi * np.arcsin(h / d_m_k) - a]))


#空对地路径损耗
β_l=1
fc=2
c=3 * (10 ** 8)
L_ploss_dmk = β_l * ((4 * np.pi * fc *d_m_k / c)**2)
σ=1
β=1
Pm=10
B=100
#给定传感器的发射功率Pm,可计算任意两点之间传输数据的时间
R_mk=B * np.log2(1+β * Pm / L_ploss_dmk * (σ**2))