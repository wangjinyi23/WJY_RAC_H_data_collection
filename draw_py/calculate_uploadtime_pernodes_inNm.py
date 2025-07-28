import numpy as np

# Parameters
distance = 50  # Distance in meters
transmit_power = 0.1  # Transmit power in watt
noise_power = 10 ** (-110 / 10) / 1000  # Noise power in watt
beta_LoS = 0  # Excessive path loss of LoS link in dB
data_packet_length = 1 * 10 ** 9  # Data packet length in bits
a = 9.61  # Environmental constant a
b = 0.16  # Environmental constant b
frequency = 2.4 * 10 ** 9  # Frequency in Hz (2.4 GHz)
transmission_rate = 1 * 10 ** 6  # Transmission rate in bps (1 Mbps)

# Calculate path loss
path_loss = 20 * np.log10(distance) + 20 * np.log10(frequency) + a - (b * np.log10(distance))

# Calculate received power
received_power = transmit_power - path_loss

# Calculate SNR
SNR = received_power - noise_power

# Calculate transmission time
transmission_time = data_packet_length / transmission_rate

print("Path Loss:", path_loss, "dB")
print("Received Power:", received_power, "dBm")
print("SNR:", SNR, "dB")
print("Transmission Time:", transmission_time, "seconds")
