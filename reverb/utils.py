import torch

###############################################################################
# UTILITIES
###############################################################################
def log_normalize(value, min, max):
  value = (torch.log10(value) - torch.log10(min)) / (torch.log10(max) - torch.log10((min)))
  return value

def log_denormalize(value, min, max):
  value = 10 ** (value * torch.log10(max) + (1.0 - value) * torch.log10(min))
  return value

def lin_normalize(value, min, max):
  value = (value - min) / (max - min)
  return value

def lin_denormalize(value, min, max):
  value = min + (max - min) * value
  return value


MIN_FREQ = 20.
MAX_FREQ = 20000.

def frequency_denormalize(f):
  min = torch.tensor(MIN_FREQ)
  max = torch.tensor(MAX_FREQ)
  return log_denormalize(f, min, max)

def frequency_normalize(f):
  min = torch.tensor(MIN_FREQ)
  max = torch.tensor(MAX_FREQ)
  return log_normalize(f, min, max)

MAX_GAIN_DB =  -1.
MIN_GAIN_DB = - 250.

def gain_denormalize(g):
  min = torch.tensor(MIN_GAIN_DB)
  max = torch.tensor(MAX_GAIN_DB)
  return lin_denormalize(g, min, max)

def gain_normalize(g):
  min = torch.tensor(MIN_GAIN_DB)
  max = torch.tensor(MAX_GAIN_DB)
  return lin_normalize(g, min, max)

MIN_Q = 0.2
MAX_Q = 2.

def q_denormalize(q):
  min = torch.tensor(MIN_Q)
  max = torch.tensor(MAX_Q)
  return log_denormalize(q, min, max)

def q_normalize(q):
  min = torch.tensor(MIN_Q)
  max = torch.tensor(MAX_Q)
  return log_normalize(q, min, max)


MIN_RT = 0.05
MAX_RT = 100.0

def rt_normalize(rt):
  min = torch.tensor(MIN_RT)
  max = torch.tensor(MAX_RT)
  return lin_normalize(rt, min, max)

def rt_denormalize(rt):
  min = torch.tensor(MIN_RT)
  max = torch.tensor(MAX_RT)
  return lin_denormalize(rt, min, max)

def convert_time_to_response(rt, delays, sample_rate):
  response_dB = (-60 * delays.to(rt.device)) / (sample_rate * rt.unsqueeze(-1))
  return response_dB

def convert_proto_gain_to_delay(gamma, delays, fs):
  gain_dB = gamma * (delays / fs)
  return gain_dB

def convert_response_to_rt(response_dB, delay, sample_rate):
  rt = (-60 * delay) / (sample_rate * response_dB)
  return rt

def check_parameter_bounds(freq, gain, q, name=""):
    """Debug utility to check if parameters are in reasonable ranges"""
    print(f"=== {name} Parameter Check ===")
    print(f"Frequencies: min={freq.min().item():.1f}, max={freq.max().item():.1f}, mean={freq.mean().item():.1f}")
    print(f"Gains: min={gain.min().item():.2f}, max={gain.max().item():.2f}, mean={gain.mean().item():.2f}")
    print(f"Q values: min={q.min().item():.3f}, max={q.max().item():.3f}, mean={q.mean().item():.3f}")
    
    # Check for problematic values
    if freq.min() < 20 or freq.max() > 20000:
        print("WARNING: Frequencies outside audible range!")
    if torch.abs(gain).max() > 24:
        print("WARNING: Extreme gain values detected!")
def is_prime(n: int) -> bool:
    """Utility to check if a number is prime."""
    if n <= 1: return False
    if n <= 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def get_next_prime(n: int) -> int:
    """Finds the next prime number greater than or equal to n."""
    prime = int(n)
    while not is_prime(prime):
        prime += 1
    return prime

def setup_fdn_delays(num_delays: int, min_ms: float, max_ms: float, sample_rate: int = 48000) -> torch.Tensor:
    """
    Calculates mutually prime delay lengths for an FDN.
    Uses logarithmic spacing to mimic natural room acoustics, 
    then snaps each length to the nearest unique prime number to avoid coloration.
    """
    min_samples = (min_ms / 1000.0) * sample_rate
    max_samples = (max_ms / 1000.0) * sample_rate

    # Logarithmic spacing for natural reflection distribution
    base = torch.linspace(0, 1, steps=num_delays)
    target_samples = min_samples * (max_samples / min_samples) ** base

    prime_lengths = []
    for target in target_samples:
        target_int = int(torch.round(target).item())
        prime = get_next_prime(target_int)
        
        # Ensure strict uniqueness (no duplicates if targets are too close)
        if prime_lengths and prime <= prime_lengths[-1]:
            prime = get_next_prime(prime_lengths[-1] + 1)
            
        prime_lengths.append(prime)

    return torch.tensor(prime_lengths, dtype=torch.int32)