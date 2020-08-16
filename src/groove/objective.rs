use crate::groove::{vars, tools};
use crate::utils_rust::transformations::{*};
use nalgebra::geometry::{Translation3, UnitQuaternion, Quaternion};
use std::cmp;
use crate::groove::vars::RelaxedIKVars;
use nalgebra::{Vector3, Isometry3, Point3};
use ncollide3d::{shape, query};

pub fn groove_loss(x_val: f64, t: f64, d: i32, c: f64, f: f64, g: i32) -> f64 {
    -( (-(x_val - t).powi(d)) / (2.0 * c.powi(2) ) ).exp() + f * (x_val - t).powi(g)
}

pub fn groove_loss_derivative(x_val: f64, t: f64, d: i32, c: f64, f: f64, g: i32) -> f64 {
    -( (-(x_val - t).powi(d)) / (2.0 * c.powi(2) ) ).exp() *  ((-d as f64 * (x_val - t)) /  (2.0 * c.powi(2))) + g as f64 * f * (x_val - t)
}

pub trait ObjectiveTrait {
    fn call(&self, x: &[f64], v: &vars::RelaxedIKVars, frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>) -> f64;
    fn call_lite(&self, x: &[f64], v: &vars::RelaxedIKVars, ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>) -> f64;
    fn gradient(&self, x: &[f64], v: &vars::RelaxedIKVars, frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>) -> (f64, Vec<f64>) {
        let mut grad: Vec<f64> = Vec::new();
        let f_0 = self.call(x, v, frames);

        for i in 0..x.len() {
            let mut x_h = x.to_vec();
            x_h[i] += 0.000000001;
            let frames_h = v.robot.get_frames_immutable(x_h.as_slice());
            let f_h = self.call(x_h.as_slice(), v, &frames_h);
            grad.push( (-f_0 + f_h) / 0.000000001);
        }

        (f_0, grad)
    }
    fn gradient_lite(&self, x: &[f64], v: &vars::RelaxedIKVars, ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>) -> (f64, Vec<f64>) {
        let mut grad: Vec<f64> = Vec::new();
        let f_0 = self.call_lite(x, v, ee_poses);

        for i in 0..x.len() {
            let mut x_h = x.to_vec();
            x_h[i] += 0.0000001;
            let ee_poses_h = v.robot.get_ee_pos_and_quat_immutable(x_h.as_slice());
            let f_h = self.call_lite(x_h.as_slice(), v, &ee_poses_h);
            grad.push( (-f_0 + f_h) / 0.0000001);
        }

        (f_0, grad)
    }
    fn gradient_type(&self) -> usize {return 1}  // manual diff = 0, finite diff = 1
}

pub struct MatchEEPosGoals {
    pub arm_idx: usize
}
impl MatchEEPosGoals {
    pub fn new(arm_idx: usize) -> Self {Self{arm_idx}}
}
impl ObjectiveTrait for MatchEEPosGoals {
    fn call(&self, x: &[f64], v: &vars::RelaxedIKVars, frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>) -> f64 {
        let last_elem = frames[self.arm_idx].0.len() - 1;
        let x_val = ( frames[self.arm_idx].0[last_elem] - v.goal_positions[self.arm_idx] ).norm();

        groove_loss(x_val, 0., 2, 0.1, 10.0, 2)
    }

    fn call_lite(&self, x: &[f64], v: &vars::RelaxedIKVars, ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>) -> f64 {
        let x_val = ( ee_poses[self.arm_idx].0 - v.goal_positions[self.arm_idx] ).norm();
        groove_loss(x_val, 0., 2, 0.1, 10.0, 2)
    }
}

pub struct MatchEEQuatGoals {
    pub arm_idx: usize
}
impl MatchEEQuatGoals {
    pub fn new(arm_idx: usize) -> Self {Self{arm_idx}}
}
impl ObjectiveTrait for MatchEEQuatGoals {
    fn call(&self, x: &[f64], v: &vars::RelaxedIKVars, frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>) -> f64 {
        let last_elem = frames[self.arm_idx].1.len() - 1;
        let tmp = Quaternion::new(-frames[self.arm_idx].1[last_elem].w, -frames[self.arm_idx].1[last_elem].i, -frames[self.arm_idx].1[last_elem].j, -frames[self.arm_idx].1[last_elem].k);
        let ee_quat2 = UnitQuaternion::from_quaternion(tmp);

        let disp = angle_between(v.goal_quats[self.arm_idx], frames[self.arm_idx].1[last_elem]);
        let disp2 = angle_between(v.goal_quats[self.arm_idx], ee_quat2);
        let x_val = disp.min(disp2);

        groove_loss(x_val, 0., 2, 0.1, 10.0, 2)
    }

    fn call_lite(&self, x: &[f64], v: &vars::RelaxedIKVars, ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>) -> f64 {
        let tmp = Quaternion::new(-ee_poses[self.arm_idx].1.w, -ee_poses[self.arm_idx].1.i, -ee_poses[self.arm_idx].1.j, -ee_poses[self.arm_idx].1.k);
        let ee_quat2 = UnitQuaternion::from_quaternion(tmp);

        let disp = angle_between(v.goal_quats[self.arm_idx], ee_poses[self.arm_idx].1);
        let disp2 = angle_between(v.goal_quats[self.arm_idx], ee_quat2);
        let x_val = disp.min(disp2);
        groove_loss(x_val, 0., 2, 0.1, 10.0, 2)
    }
}

pub struct NNSelfCollision;
impl ObjectiveTrait for NNSelfCollision {
    fn call(&self, x: &[f64], v: &vars::RelaxedIKVars, frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>) -> f64 {
        let mut x_val = v.collision_nn.predict(&x.to_vec());
        groove_loss(x_val, 0., 2, 2.1, 0.0002, 4)
    }

    fn call_lite(&self, x: &[f64], v: &vars::RelaxedIKVars, ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>) -> f64 {
        let mut x_val = v.collision_nn.predict(&x.to_vec());
        groove_loss(x_val, 0., 2, 2.1, 0.0002, 4)
    }

    fn gradient(&self, x: &[f64], v: &vars::RelaxedIKVars, frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>) -> (f64, Vec<f64>) {
        let (x_val, mut grad) = v.collision_nn.gradient(&x.to_vec());
        let g_prime = groove_loss_derivative(x_val, 0., 2, 2.1, 0.0002, 4);
        for i in 0..grad.len() {
            grad[i] *= g_prime;
        }
        (x_val, grad)
    }

    fn gradient_lite(&self, x: &[f64], v: &vars::RelaxedIKVars, ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>) -> (f64, Vec<f64>) {
        let (x_val, mut grad) = v.collision_nn.gradient(&x.to_vec());
        let g_prime = groove_loss_derivative(x_val, 0., 2, 2.1, 0.0002, 4);
        for i in 0..grad.len() {
            grad[i] *= g_prime;
        }
        (x_val, grad)
    }

    fn gradient_type(&self) -> usize {return 0}
}

pub struct EnvCollision {
    pub arm_idx: usize
}
impl EnvCollision {
    pub fn new(arm_idx: usize) -> Self {Self{arm_idx}}
}
impl ObjectiveTrait for EnvCollision {
    fn call(&self, x: &[f64], v: &vars::RelaxedIKVars, frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>) -> f64 {
        let link_radius = v.env_collision.robot_link_radius;
        let penalty_cutoff: f64 = link_radius / 2.0;
        // println!("Penalty: {}", penalty_cutoff);
        let a = 0.005 * (penalty_cutoff.powi(10));
        let planes = &v.env_collision.cuboids;
        let spheres = &v.env_collision.spheres;
        let mut sum_max: f64 = 0.0;
        let last_elem = frames[self.arm_idx].0.len() - 1;
        // println!("Frames: {:?}", frames[self.arm_idx]);
        for i in 0..planes.len() {
            let plane = shape::Cuboid::new(Vector3::new(planes[i].x_halflength, planes[i].y_halflength, planes[i].z_halflength));
            let plane_ts = Translation3::new(planes[i].tx, planes[i].ty, planes[i].tz);
            let plane_rot = UnitQuaternion::from_euler_angles(planes[i].rx, planes[i].ry, planes[i].rz);
            let plane_pos = Isometry3::from_parts(plane_ts, plane_rot);
            let mut sum: f64 = 0.0;
            for j in 0..last_elem {
                let start_pt = Point3::from(frames[self.arm_idx].0[j]);
                let end_pt = Point3::from(frames[self.arm_idx].0[j + 1]);
                let segment = shape::Segment::new(start_pt, end_pt);
                let segment_pos = nalgebra::one();
                let dis = query::distance(&plane_pos, &plane, &segment_pos, &segment) - link_radius;
                // println!("Link: {}, Distance: {:?}", j, dis);
                sum += a / dis.powi(10); 
            }
            sum_max = sum_max.max(sum);
        }

        for i in 0..spheres.len() {
            let collision_pt = nalgebra::Point3::new(spheres[i].tx, spheres[i].ty, spheres[i].tz);
            let obs_r = spheres[i].radius;
            // println!("Point: {:?}, radius: {}", collision_pt, obs_r);
            let mut sum: f64 = 0.0;
            for j in 0..last_elem {
                let start_pt = nalgebra::Point3::from(frames[self.arm_idx].0[j]);
                let end_pt = nalgebra::Point3::from(frames[self.arm_idx].0[j + 1]);
                let link_len = nalgebra::distance(&start_pt, &end_pt);
                let dot = (end_pt - start_pt).dot(&(collision_pt - start_pt));
                let param = dot / link_len.powi(2);
                let mut dis;
                if param <= 0.0 {
                    dis = nalgebra::distance(&start_pt, &collision_pt);
                } else if param >= 1.0 {
                    dis = nalgebra::distance(&end_pt, &collision_pt);
                } else {
                    let cross_product = (start_pt - end_pt).cross(&(start_pt - collision_pt));
                    dis = nalgebra::Matrix::magnitude(&cross_product) / link_len;
                }
                dis = dis - obs_r - link_radius;
                // println!("Link: {} Start: {:?} End: {:?} Length: {:?} Param: {:?} Distance: {:?}", j, start_pt, end_pt, link_len, param, dis);
                sum += a / dis.powi(10); 
            }
            // println!("Obstacle: {}, Sum: {}", i, sum);
            sum_max = sum_max.max(sum);
        }
        
        // println!("Sum Max: {}", sum_max);
        groove_loss(sum_max, 0., 2, 2.1, 0.0002, 4)
    }

    fn call_lite(&self, x: &[f64], v: &vars::RelaxedIKVars, ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>) -> f64 {
        let x_val = 1.0; 
        groove_loss(x_val, 0., 2, 2.1, 0.0002, 4)
    }
}

pub struct JointLimits;
impl ObjectiveTrait for JointLimits {
    fn call(&self, x: &[f64], v: &vars::RelaxedIKVars, frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>) -> f64 {
        let mut sum = 0.0;
        let penalty_cutoff: f64 = 0.9;
        let a = 0.05 / (penalty_cutoff.powi(50));
        for i in 0..v.robot.num_dof {
            let l = v.robot.bounds[i][0];
            let u = v.robot.bounds[i][1];
            let r = (x[i] - l) / (u - l);
            let n = 2.0 * (r - 0.5);
            sum += a*n.powf(50.);
        }
        groove_loss(sum, 0.0, 2, 0.32950, 0.1, 2)
    }

    fn call_lite(&self, x: &[f64], v: &vars::RelaxedIKVars, ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>) -> f64 {
        let mut sum = 0.0;
        let penalty_cutoff: f64 = 0.85;
        let a = 0.05 / (penalty_cutoff.powi(50));
        for i in 0..v.robot.num_dof {
            let l = v.robot.bounds[i][0];
            let u = v.robot.bounds[i][1];
            let r = (x[i] - l) / (u - l);
            let n = 2.0 * (r - 0.5);
            sum += a*n.powi(50);
        }
        groove_loss(sum, 0.0, 2, 0.32950, 0.1, 2)
    }
}

pub struct MinimizeVelocity;
impl ObjectiveTrait for MinimizeVelocity {
    fn call(&self, x: &[f64], v: &vars::RelaxedIKVars, frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>) -> f64 {
        let mut x_val = 0.0;
        for i in 0..x.len() {
           x_val += (x[i] - v.xopt[i]).powi(2);
        }
        x_val = x_val.sqrt();
        groove_loss(x_val, 0.0, 2, 0.1, 10.0, 2)
    }

    fn call_lite(&self, x: &[f64], v: &vars::RelaxedIKVars, ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>) -> f64 {
        let mut x_val = 0.0;
        for i in 0..x.len() {
           x_val += (x[i] - v.xopt[i]).powi(2);
        }
        x_val = x_val.sqrt();
        groove_loss(x_val, 0.0, 2, 0.1, 10.0, 2)
    }

}

pub struct MinimizeAcceleration;
impl ObjectiveTrait for MinimizeAcceleration {
    fn call(&self, x: &[f64], v: &vars::RelaxedIKVars, frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>) -> f64 {
        let mut x_val = 0.0;
        for i in 0..x.len() {
            let v1 = x[i] - v.xopt[i];
            let v2 = v.xopt[i] - v.prev_state[i];
            x_val += (v1 - v2).powi(2);
        }
        x_val = x_val.sqrt();
        groove_loss(x_val, 0.0, 2, 0.1, 10.0, 2)
    }

    fn call_lite(&self, x: &[f64], v: &vars::RelaxedIKVars, ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>) -> f64 {
        let mut x_val = 0.0;
        for i in 0..x.len() {
            let v1 = x[i] - v.xopt[i];
            let v2 = v.xopt[i] - v.prev_state[i];
            x_val += (v1 - v2).powi(2);
        }
        x_val = x_val.sqrt();
        groove_loss(x_val, 0.0, 2, 0.1, 10.0, 2)
    }
}

pub struct MinimizeJerk;
impl ObjectiveTrait for MinimizeJerk {
    fn call(&self, x: &[f64], v: &vars::RelaxedIKVars, frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>) -> f64 {
        let mut x_val = 0.0;
        for i in 0..x.len() {
            let v1 = x[i] - v.xopt[i];
            let v2 = v.xopt[i] - v.prev_state[i];
            let v3 = v.prev_state[i] - v.prev_state2[i];
            let a1 = v1 - v2;
            let a2 = v2 - v3;
            x_val += (a1 - a2).powi(2);
        }
        x_val = x_val.sqrt();
        groove_loss(x_val, 0.0, 2, 0.1, 10.0, 2)
    }

    fn call_lite(&self, x: &[f64], v: &vars::RelaxedIKVars, ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>) -> f64 {
        let mut x_val = 0.0;
        for i in 0..x.len() {
            let v1 = x[i] - v.xopt[i];
            let v2 = v.xopt[i] - v.prev_state[i];
            let v3 = v.prev_state[i] - v.prev_state2[i];
            let a1 = v1 - v2;
            let a2 = v2 - v3;
            x_val += (a1 - a2).powi(2);
        }
        x_val = x_val.sqrt();
        groove_loss(x_val, 0.0, 2, 0.1, 10.0, 2)
    }
}