from sparse_grid import SparseGrid


if __name__ == "__main__":
    # Create a sparse grid object
    sparse_grid = SparseGrid(
        size=5, max_steps=50, vector_state=False, render=True, reward_type="sparse"
    )

    # Printing details
    print(sparse_grid.action_space_description)
    print(sparse_grid.reward_space_description)

    # Reset the environment
    obs = sparse_grid.reset()

    for _ in range(sparse_grid.max_steps):
        # Take a random action
        action = sparse_grid.random_action()
        new_obs = sparse_grid.step(action)

        # Extract information from the new observation
        observation = new_obs[0]["observation"]
        achieved_goal = new_obs[0]["achieved_goal"]
        desired_goal = new_obs[0]["desired_goal"]
        done = new_obs[-1]
        reward = new_obs[-2]

        # Print the information
        print(f"Reward: {reward}")
