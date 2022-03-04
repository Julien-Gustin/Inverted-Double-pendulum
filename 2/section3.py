import cv2
from python.gui import save_caronthehill_image, CANVAS_HEIGHT, CANVAS_WIDTH
from python.domain import State, CarOnTheHillDomain
from python.simulation import Simulation
from python.policy import AlwaysAcceleratePolicy, RandomActionPolicy

from python.constants import TIME_STEP

def make_video(output_video: str, trajectory: list):
    fps = 1/TIME_STEP
    video = cv2.VideoWriter(output_video, 0, fps, (CANVAS_WIDTH, CANVAS_HEIGHT))
    for t in trajectory:
        prev_state, action, reward, new_state = t
        save_caronthehill_image(prev_state.p, prev_state.s, "images/frame.png")
        image = cv2.imread("images/frame.png")
        video.write(image)
    save_caronthehill_image(0, 0, "images/frame.png", close=True)
    video.release()

if __name__=="__main__":
    steps = 600
    domain = CarOnTheHillDomain()
    initial_state = State(0, 0)

    policy = RandomActionPolicy()
    simulation = Simulation(domain, policy, initial_state, remember_trajectory=True)
    simulation.simulate(steps)
    make_video("video/animation_random.avi", simulation.get_trajectory())

    policy = AlwaysAcceleratePolicy()
    simulation = Simulation(domain, policy, initial_state, remember_trajectory=True)
    simulation.simulate(steps)
    make_video("video/animation_always_accelerate.avi", simulation.get_trajectory())