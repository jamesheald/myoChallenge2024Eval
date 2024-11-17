import mujoco
import numpy as np

def check_joint_limits(model, q, ids):
        """Check if the joints is under or above its limits"""
        for i in range(len(q)):
            id = ids[i]
            q[i] = max(
                model.jnt_range[id][0], min(q[i], model.jnt_range[id][1])
            )

class GradientDescentIK:
    def __init__(self, model, data, step_size, tol, alpha, jacp, jacr):
        self.model = model
        self.data = data
        self.step_size = step_size
        self.tol = tol
        self.alpha = alpha
        self.jacp = jacp
        self.jacr = jacr

    # Gradient Descent pseudocode implementation
    def calculate(self, goal, init_q, body_id, prosthesis_ids, max_steps=5000):
        """Calculate the desire joints angles for goal"""
        assert len(init_q) == len(prosthesis_ids)
        self.data.qpos[prosthesis_ids] = init_q
        mujoco.mj_forward(self.model, self.data)
        current_pose = self.data.body(body_id).xpos
        error = np.subtract(goal, current_pose)
        step = 0
        while np.linalg.norm(error) >= self.tol:
            # print(f"Step: {step} | Error: {np.linalg.norm(error)}")
            step += 1
            # calculate jacobian
            mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, goal, body_id)
            # calculate gradient
            grad = self.alpha * self.jacp.T @ error
            # compute next step
            self.data.qpos[prosthesis_ids] += self.step_size * grad[prosthesis_ids]
            # check joint limits
            check_joint_limits(self.model, self.data.qpos[prosthesis_ids], prosthesis_ids)
            # compute forward kinematics
            mujoco.mj_forward(self.model, self.data)
            # calculate new error
            error = np.subtract(goal, self.data.body(body_id).xpos)

            if step >= max_steps:
                break

        return self.data.qpos[prosthesis_ids].copy()