from typing import Tuple, List, Union, Dict
from dataclasses import dataclass

import numpy as np
import pygame
from time import sleep


@dataclass
class Unit:
    def __init__(self, name: str, id: int) -> None:
        self._position: Union[None, Tuple[int, int]] = None
        self._id = id
        self._name = name

    @property
    def position(self) -> np.ndarray:
        return np.hstack(self._position)

    @position.setter
    def position(self, pos) -> None:
        self._position = pos

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> Dict[str, np.ndarray]:
        return {self.name: np.hstack([self.id, list(self.position)])}

    @property
    def vector_state(self) -> Dict[str, np.ndarray]:
        return np.hstack([self.id, list(self.position)])


@dataclass
class Taxi(Unit):
    def __init__(self, name: str, id: int) -> None:
        super().__init__(name, id)

        self._passenger: Union[None, Unit] = None

    def pick_up(self, unit: Unit) -> None:
        self._passenger = unit

    def drop_off(self) -> None:
        self._passenger = None

    @property
    def passenger(self) -> Union[None, Unit]:
        return self._passenger

    @passenger.setter
    def passenger(self, passenger: Unit) -> None:
        self._passenger = passenger

    @property
    def state(self) -> List[Dict[str, np.ndarray]]:
        if self.passenger is None:
            passenger_value = np.hstack([-1, -1, -1])
        else:
            passenger_value = np.hstack(
                [self.passenger.id, list(self.passenger.position)]
            )

        return [
            {self.name: np.hstack([self.id, list(self.position)])},
            {"P": passenger_value},
        ]

    @property
    def vector_state(self) -> np.ndarray:
        if self.passenger is None:
            passenger_value = np.hstack([-1, -1, -1])
        else:
            passenger_value = np.hstack(
                [self.passenger.id, list(self.passenger.position)]
            )

        return np.hstack([self.id, list(self.position), passenger_value])


class Renderer:
    def __init__(self, grid_size: int) -> None:
        self._grid_size = grid_size
        self.colors = {
            "R": (255, 0, 0),
            "G": (0, 255, 0),
            "B": (0, 0, 255),
            "T": (255, 255, 0),
            "F": (255, 255, 255),
        }

        self.width, self.height = 500, 500

        area = self.width * self.height
        self.unit_size = np.sqrt(area / (grid_size**2))

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Grid")

    @property
    def display_transformation(self) -> np.ndarray:
        return np.array([[1, 0, 0], [0, -1, self.height - self.unit_size], [0, 0, 1]])

    def compute_display_position(self, x: int, y: int) -> Tuple[int, int]:
        u, v, _ = self.display_transformation @ np.array([x, y, 1])
        return u, v

    def render(self, state: Dict[str, np.ndarray]) -> None:
        self.screen.fill((0, 0, 0))

        for unit, position in state.items():
            color = self.colors.get(unit)
            if color is None:
                continue
            x = position[1] * self.unit_size
            y = position[2] * self.unit_size
            x, y = self.compute_display_position(x, y)
            if unit == "F":
                pygame.draw.rect(
                    self.screen,
                    color,
                    pygame.Rect(x, y, self.unit_size, self.unit_size),
                    2,
                )
            else:
                pygame.draw.rect(
                    self.screen,
                    color,
                    pygame.Rect(x, y, self.unit_size, self.unit_size),
                )

        # render special case for taxi with passenger
        position = state["P"]
        color = None
        if position[0] == 100:
            color = self.colors.get("R")
        elif position[0] == 200:
            color = self.colors.get("G")
        elif position[0] == 300:
            color = self.colors.get("B")
        if color is not None:
            x = position[1] * self.unit_size
            y = position[2] * self.unit_size
            x, y = self.compute_display_position(x, y)
            pygame.draw.rect(
                self.screen, color, pygame.Rect(x, y, self.unit_size, self.unit_size), 2
            )

        pygame.display.update()
        sleep(0.05)

    def reset(self) -> None:
        for _ in range(10):
            random_color = np.random.randint(0, 255, size=3)
            self.screen.fill(random_color)
            pygame.display.update()
            sleep(0.02)


class SparseGrid:
    def __init__(
        self,
        size: int = 5,
        max_steps: int = 25,
        vector_state: bool = False,
        render: bool = True,
        reward_type: str = "sparse",
    ):
        super().__init__()

        if reward_type in ["dense", "sparse"]:
            self._reward_type = reward_type
        else:
            raise ValueError("Reward type options must be in {'dense', 'sparse'}")

        size = np.maximum(4, size)
        self._size = (size, size)

        self._red = Unit("R", 100)
        self._green = Unit("G", 200)
        self._blue = Unit("B", 300)

        self._taxi = Taxi("T", 400)
        self._destination = Unit("D", 500)

        self.max_steps = max_steps
        self.current_step: int = 0

        self._enable_vector_state = vector_state

        self._render = render
        if render:
            self._renderer = Renderer(size)

        # Reset for intial setting
        self.reset()

        self.sample_action = self.random_action()

    @property
    def max_action(self) -> int:
        return 6

    @property
    def min_action(self) -> int:
        return 0

    @property
    def max_reward(self) -> float:
        return 1.0

    @property
    def min_reward(self) -> float:
        return -1.0

    @property
    def action_space_description(self) -> str:
        return f"action ∈ ℤ= {(self.min_action, self.max_action)}"

    @property
    def reward_space_description(self) -> str:
        return f"reward ∈ ℝ = {(self.min_reward, self.max_reward)}"

    def step(
        self, action: int
    ) -> Tuple[Union[Dict[str, np.ndarray], np.ndarray], float, bool]:
        if action == 0:  # do nothing
            pass

        elif action == 1:  # move up
            self._taxi.position = self._update_taxi_position((0, 1))

        elif action == 2:  # move down
            self._taxi.position = self._update_taxi_position((0, -1))

        elif action == 3:  # move right
            self._taxi.position = self._update_taxi_position((1, 0))

        elif action == 4:  # move left
            self._taxi.position = self._update_taxi_position((-1, 0))

        elif action == 5:  # Pickup
            self._pickup()

        elif action == 6:  # Drop
            self._drop()

        self._set_state()
        reward = self.compute_reward()
        done_flag = self.compute_done(reward)
        self.current_step += 1

        self.render()

        return self.state(), reward, done_flag

    def state(self) -> Union[List[Dict[str, np.ndarray]], np.ndarray]:
        return self._get_state(self._enable_vector_state)

    def _update_taxi_position(self, position_step: Tuple[int, int]) -> Tuple[int, int]:
        updated_position: np.ndarray = np.hstack(self._taxi.position) + np.hstack(
            position_step
        )
        updated_position[0] = updated_position[0].clip(0, self._size[0] - 1)
        updated_position[1] = updated_position[1].clip(0, self._size[1] - 1)

        return (updated_position[0], updated_position[1])

    def _drop(self) -> bool:
        if self._taxi.passenger is not None:
            self._taxi.passenger.position = self._taxi.position
            self._taxi.passenger = None

    def _pickup(self) -> None:
        if self._red.position.tolist() == self._taxi.position.tolist():
            self._taxi.passenger = self._red

        elif self._blue.position.tolist() == self._taxi.position.tolist():
            self._taxi.passenger = self._blue

        elif self._green.position.tolist() == self._taxi.position.tolist():
            self._taxi.passenger = self._green

    @staticmethod
    def random_action() -> int:
        return np.random.randint(0, 7)

    def reset(self) -> np.ndarray:
        # Generate random positions
        prefixed_taxi_pos = (self._size[0] - 1, self._size[1] - 1)
        prefixed_destination_pos = (0, 0)
        random_positions: List[Tuple[int, int]] = []
        random_positions.append(prefixed_taxi_pos)
        random_positions.append(prefixed_destination_pos)

        # Fix taxi and destination postion
        while len(random_positions) != 5:
            random_position = (
                np.random.randint(0, self._size[0] - 1),
                np.random.randint(0, self._size[1] - 1),
            )
            if random_position not in random_positions:
                random_positions.append(random_position)

        # Assign random positions
        self._red.position = random_positions[2]
        self._green.position = random_positions[3]
        self._blue.position = random_positions[4]
        self._taxi.position = prefixed_taxi_pos
        self._destination.position = prefixed_destination_pos

        # Generate a random goal
        self._goal_candidate: Unit = np.random.choice(
            [self._red, self._green, self._blue]
        )

        # Detailed final goal data
        self._goal = {
            "F": np.hstack([self._goal_candidate.id, list(self._destination.position)])
        }

        # Clear flags
        self.current_step = 0

        # Handle the renders
        if self._render:
            self._renderer.reset()
            self.render()

        return self.state()

    def render(self) -> None:
        if self._render:
            self._renderer.render(self._get_state(vector_state=False, detailed=True))

    def compute_reward(self) -> float:
        goal_passenger_id = self._goal_candidate.id

        # Return 1.0 if the puzzle is solved!
        for passenger in [self._red, self._green, self._blue]:
            if (
                passenger.id == goal_passenger_id
                and np.linalg.norm(self._destination.position - passenger.position) == 0
                and self._taxi.passenger is None
            ):
                return 1.0

        if self._reward_type == "sparse":
            return -1.0

        else:
            state = self._get_state(False)["observation"]

            taxi_passenger_distance = np.linalg.norm(state[-2:])
            goal_passenger_distance = np.linalg.norm(state[-4:-2])

            return -1.0 * (taxi_passenger_distance + goal_passenger_distance)

    def compute_done(self, reward: float) -> bool:
        # Check if passengers overlap
        current_position_list = [
            self._red.position,
            self._blue.position,
            self._green.position,
        ]
        stack_current_position = np.vstack(current_position_list)
        unique_current_position = np.unique(stack_current_position, axis=0)

        if self.current_step == self.max_steps - 1:
            return True

        elif reward == 1.0 and self._reward_type == "sparse":
            return True

        if (
            self._goal_candidate.position.tolist()
            == self._destination.position.tolist()
            and self._reward_type == "dense"
        ):
            return True

        elif stack_current_position.shape[0] != unique_current_position.shape[0]:
            return False

        else:
            return False

    def _set_state(self) -> None:
        if self._taxi.passenger is not None:
            self._taxi.passenger.position = self._taxi.position

    def _get_state(
        self, vector_state: bool, detailed: bool = False
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        red_position = self._red.position
        green_position = self._green.position
        blue_position = self._blue.position
        taxi_position = self._taxi.position

        target_passenger_position = self._goal_candidate.position
        taxi_target_rel_position = abs(target_passenger_position - taxi_position)

        goal_position = self._destination.position
        goal_target_passenger_rel_pos = abs(goal_position - target_passenger_position)

        state_dict: List[Dict[str, np.ndarray]] = [
            self._red.state,
            self._green.state,
            self._blue.state,
            self._taxi.state[0],
            self._taxi.state[1],
            self._destination.state,
            self._goal,
        ]

        if detailed:
            return {k: v for d in state_dict for k, v in d.items()}

        if not vector_state:
            return {
                "observation": np.hstack(
                    [
                        red_position,
                        green_position,
                        blue_position,
                        taxi_position,
                        taxi_target_rel_position,
                        goal_target_passenger_rel_pos,
                    ]
                ),
                "achieved_goal": np.hstack([target_passenger_position]),
                "desired_goal": np.hstack([goal_position]),
            }
        else:
            return np.hstack(
                [
                    red_position,
                    green_position,
                    blue_position,
                    taxi_position,
                    target_passenger_position,
                    taxi_target_rel_position,
                    target_passenger_position,
                    goal_position,
                    goal_target_passenger_rel_pos,
                ]
            )
