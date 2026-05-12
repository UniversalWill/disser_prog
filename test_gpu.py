#!/usr/bin/env python3
"""Test GPU access with JAX + SBX (Stable Baselines Jax)."""

import os
import sys
import time
import subprocess
import platform

SEP = "=" * 60


def check_cuda_driver():
    """Check NVIDIA driver / CUDA version via nvidia-smi."""
    print("\n" + SEP)
    print("CUDA DRIVER CHECK")
    print(SEP)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,vbios_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print(f"  GPU: {result.stdout.strip()}")
        else:
            print("  nvidia-smi returned non-zero")
    except FileNotFoundError:
        print("  nvidia-smi not found — no NVIDIA driver?")
    except Exception as e:
        print(f"  nvidia-smi error: {e}")


def test_jax_gpu():
    """Verify JAX sees the GPU and can compute on it."""
    print("\n" + SEP)
    print("JAX GPU DETECTION")
    print(SEP)

    try:
        import jax
        print(f"  JAX version: {jax.__version__}")
        print(f"  JAX platform: {jax.default_backend()}")
    except ImportError as e:
        print(f"  FAILED: Cannot import jax — {e}")
        return False

    devices = jax.devices()
    gpus = jax.devices("gpu")
    cpus = jax.devices("cpu")

    print(f"  All devices ({len(devices)}): {[str(d) for d in devices]}")
    print(f"  GPU devices ({len(gpus)}): {[str(d) for d in gpus]}")
    print(f"  CPU devices ({len(cpus)}): {[str(d) for d in cpus]}")

    if not gpus:
        print("\n  FAILED: No GPU devices found by JAX!")
        print("  Check: pip install jax[cuda13] and nvidia-smi")
        return False

    # Quick compute test on GPU
    print("\n  Running GPU compute test...")
    try:
        import jax.numpy as jnp

        with jax.default_device(gpus[0]):
            a = jnp.ones((1000, 1000))
            b = jnp.ones((1000, 1000))
            t0 = time.time()
            c = jnp.dot(a, b).block_until_ready()
            t1 = time.time()

        print(f"  Matrix 1000x1000 dot product: {t1 - t0:.4f}s (GPU)")
        print(f"  Result shape: {c.shape}, sum={float(jnp.sum(c)):.1f}")
        print("  GPU compute: OK")
        return True
    except Exception as e:
        print(f"  GPU compute FAILED: {e}")
        return False


def test_sbx_ppo():
    """Create and train an SBX PPO model on CartPole-v1."""
    print("\n" + SEP)
    print("SBX PPO TRAINING TEST")
    print(SEP)

    try:
        import jax
        from sbx import PPO
        import gymnasium as gym
    except ImportError as e:
        print(f"  FAILED: Cannot import sbx/gymnasium — {e}")
        return False

    gpus = jax.devices("gpu")
    if not gpus:
        print("  Skipping: no GPU available")
        return False

    print(f"  Using device: {gpus[0]}")
    print(f"  Environment: CartPole-v1")

    with jax.default_device(gpus[0]):
        # Create simple env for quick test
        env = gym.make("CartPole-v1")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")

        # Create PPO model (same algo as existing project uses via SB3)
        print("\n  Creating SBX PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            n_steps=128,
            batch_size=64,
            n_epochs=4,
            learning_rate=3e-4,
            policy_kwargs=dict(net_arch=[64, 64]),
            tensorboard_log=None,
            seed=42,
        )
        print("  Model created successfully")

        # Train for a small number of timesteps
        total_timesteps = 2048
        print(f"\n  Training for {total_timesteps} timesteps...")
        t0 = time.time()
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
        t1 = time.time()
        elapsed = t1 - t0

        print(f"\n  Training completed in {elapsed:.2f}s")
        print(f"  Speed: {total_timesteps / elapsed:.0f} steps/sec")

        # Quick evaluation
        print("\n  Running evaluation episodes...")
        num_episodes = 5
        total_reward = 0.0
        for ep in range(num_episodes):
            obs, _ = env.reset(seed=42 + ep)
            done = False
            truncated = False
            ep_reward = 0.0
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(int(action))
                ep_reward += reward
            total_reward += ep_reward
            print(f"    Episode {ep + 1}: reward = {ep_reward:.0f}")

        print(f"  Average reward over {num_episodes} episodes: {total_reward / num_episodes:.1f}")
        env.close()

    print("\n  SBX PPO test: OK")
    return True


def test_jax_vs_numpy():
    """Compare JAX GPU vs NumPy CPU compute speed."""
    print("\n" + SEP)
    print("JAX GPU vs NumPy CPU SPEED COMPARISON")
    print(SEP)

    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
    except ImportError:
        print("  Skipping: missing dependencies")
        return

    gpus = jax.devices("gpu")
    if not gpus:
        print("  Skipping: no GPU available")
        return

    size = 2000

    # NumPy CPU
    a_cpu = np.random.randn(size, size).astype(np.float32)
    b_cpu = np.random.randn(size, size).astype(np.float32)
    t0 = time.time()
    c_cpu = np.dot(a_cpu, b_cpu)
    t1 = time.time()
    cpu_time = t1 - t0

    # JAX GPU
    with jax.default_device(gpus[0]):
        a_gpu = jnp.array(a_cpu)
        b_gpu = jnp.array(b_cpu)
        jit_dot = jax.jit(lambda x, y: jnp.dot(x, y))
        # Warmup
        _ = jit_dot(a_gpu, b_gpu).block_until_ready()
        t0 = time.time()
        c_gpu = jit_dot(a_gpu, b_gpu).block_until_ready()
        t1 = time.time()
    gpu_time = t1 - t0

    print(f"  Matrix multiply {size}x{size} (32-bit):")
    print(f"    NumPy (CPU):     {cpu_time:.4f}s")
    print(f"    JAX (GPU, JIT):  {gpu_time:.4f}s")
    if cpu_time > 0:
        print(f"    Speedup:         {cpu_time / gpu_time:.1f}x")


def main():
    print("=" * 60)
    print("GPU ACCESS TEST — JAX + SBX (Stable Baselines Jax)")
    print("=" * 60)
    print(f"  Python:   {sys.version}")
    print(f"  Platform: {platform.platform()}")

    check_cuda_driver()

    jax_ok = test_jax_gpu()
    if not jax_ok:
        print("\n" + "=" * 60)
        print("RESULT: JAX GPU detection FAILED")
        print("=" * 60)
        return 1

    test_jax_vs_numpy()
    sbx_ok = test_sbx_ppo()

    print("\n" + "=" * 60)
    if sbx_ok:
        print("RESULT: All tests PASSED — JAX + SBX GPU is working")
    else:
        print("RESULT: JAX GPU OK, but SBX test had issues")
    print("=" * 60)
    return 0 if sbx_ok else 1


if __name__ == "__main__":
    sys.exit(main())
