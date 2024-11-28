import gymnasium as gym
import matplotlib.pyplot as plt
import imageio
import os

# Create a directory to store frames
if not os.path.exists("frames"):
    os.makedirs("frames")

env = gym.make("LunarLander-v2", render_mode="rgb_array")
observation, info = env.reset()

frames = []  # List to store all the frames

for i in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

    # Render and collect the frame
    frame = env.render()  # This returns an image array
    frames.append(frame)  # Add each frame to the list

env.close()

# Save the frames as a GIF
imageio.mimsave("lunar_lander.gif", frames, fps=30)  # fps=30 for smooth playback