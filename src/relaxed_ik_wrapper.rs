use crate::relaxed_ik::{RelaxedIK, Opt};
use std::sync::{Arc, Mutex};
use nalgebra::{Unit, Vector3, Vector6, UnitQuaternion, Quaternion,Translation3, Isometry3};
use std::os::raw::{*};
use std::str;
use crate::utils_rust::file_utils::{*};

// http://jakegoulding.com/rust-ffi-omnibus/objects/
#[no_mangle]
pub unsafe extern "C" fn relaxed_ik_new(path_to_setting: *const c_char) -> *mut RelaxedIK {
    if path_to_setting.is_null() 
    { 
        let path_to_src = get_path_to_src();
        let default_path_to_setting = path_to_src +  "configs/settings.yaml";
        return Box::into_raw(Box::new(RelaxedIK::load_settings(default_path_to_setting.as_str())))
    }
    let c_str = std::ffi::CStr::from_ptr(path_to_setting);
    let path_to_setting_str = c_str.to_str().expect("Not a valid UTF-8 string");

    Box::into_raw(Box::new(RelaxedIK::load_settings(path_to_setting_str)))
}

#[no_mangle] 
pub unsafe extern "C" fn relaxed_ik_free(ptr: *mut RelaxedIK) {
    if ptr.is_null() { return }
    Box::from_raw(ptr);
}

#[no_mangle]
pub unsafe extern "C" fn reset(ptr: *mut RelaxedIK, joint_state: *const c_double, joint_state_length: c_int) {
    let relaxed_ik = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };

    let x_slice: &[c_double] = std::slice::from_raw_parts(joint_state, joint_state_length as usize);
    let x_vec = x_slice.to_vec();
    relaxed_ik.reset(x_vec);
}

#[no_mangle]
pub unsafe extern "C" fn solve_position(ptr: *mut RelaxedIK, pos_goals: *const c_double, pos_length: c_int, 
    quat_goals: *const c_double, quat_length: c_int,
    tolerance: *const c_double, tolerance_length: c_int) -> Opt {

    let relaxed_ik = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };
    
    assert!(!pos_goals.is_null(), "Null pointer for pos goals!");
    assert!(!quat_goals.is_null(), "Null pointer for quat goals!");
    assert!(!tolerance.is_null(), "Null pointer for tolerance!"); 

    assert!(pos_length as usize == relaxed_ik.vars.robot.num_chains * 3 , 
        "Pos vels are expected to have {} numbers, but got {}", 
        relaxed_ik.vars.robot.num_chains * 3, pos_length);
    assert!(quat_length as usize == relaxed_ik.vars.robot.num_chains * 4,
        "Rot vels are expected to have {} numbers, but got {}", 
        relaxed_ik.vars.robot.num_chains * 4, quat_length);
    assert!(tolerance_length as usize == relaxed_ik.vars.robot.num_chains * 6, 
        "Tolerance are expected to have {} numbers, but got {}", 
        relaxed_ik.vars.robot.num_chains * 6, tolerance_length);

    let pos_slice: &[c_double] = std::slice::from_raw_parts(pos_goals, pos_length as usize);
    let quat_slice: &[c_double] = std::slice::from_raw_parts(quat_goals, quat_length as usize);
    let tolerance_slice: &[c_double] = std::slice::from_raw_parts(tolerance, tolerance_length as usize);

    let pos_vec = pos_slice.to_vec();
    let quat_vec = quat_slice.to_vec();
    let tolerance_vec = tolerance_slice.to_vec();

    let ja = solve_position_helper(relaxed_ik, pos_vec, quat_vec, tolerance_vec);

    let ptr = ja.as_ptr();
    let len = ja.len();
    std::mem::forget(ja);
    Opt {data: ptr, length: len as c_int}
}


#[no_mangle]
pub unsafe extern "C" fn solve_velocity(ptr: *mut RelaxedIK, pos_vels: *const c_double, pos_length: c_int, 
    rot_vels: *const c_double, rot_length: c_int,
    tolerance: *const c_double, tolerance_length: c_int) -> Opt {
    
    let relaxed_ik = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };
    
    assert!(!pos_vels.is_null(), "Null pointer for pos vels!");
    assert!(!rot_vels.is_null(), "Null pointer for rot vels!");
    assert!(!tolerance.is_null(), "Null pointer for tolerance!"); 

    assert!(pos_length as usize == relaxed_ik.vars.robot.num_chains * 3 , 
        "Pos vels are expected to have {} numbers, but got {}", 
        relaxed_ik.vars.robot.num_chains * 3, pos_length);
    assert!(rot_length as usize == relaxed_ik.vars.robot.num_chains * 3,
        "Rot vels are expected to have {} numbers, but got {}", 
        relaxed_ik.vars.robot.num_chains * 3, rot_length);
    assert!(tolerance_length as usize == relaxed_ik.vars.robot.num_chains * 6, 
        "Tolerance are expected to have {} numbers, but got {}", 
        relaxed_ik.vars.robot.num_chains * 6, tolerance_length);

    let pos_slice: &[c_double] = std::slice::from_raw_parts(pos_vels, pos_length as usize);
    let rot_slice: &[c_double] = std::slice::from_raw_parts(rot_vels, rot_length as usize);
    let tolerance_slice: &[c_double] = std::slice::from_raw_parts(tolerance, tolerance_length as usize);

    let pos_vec = pos_slice.to_vec();
    let rot_vec = rot_slice.to_vec();
    let tolerance_vec = tolerance_slice.to_vec();

    let ja = solve_velocity_helper(relaxed_ik, pos_vec, rot_vec, tolerance_vec);

    let ptr = ja.as_ptr();
    let len = ja.len();
    std::mem::forget(ja);
    Opt {data: ptr, length: len as c_int}
}

#[no_mangle]
pub unsafe extern "C" fn get_ee_positions(ptr: *mut RelaxedIK) -> Opt {
    let relaxed_ik = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };

    let mut positions = Vec::new();
    for i in 0..relaxed_ik.vars.goal_positions.len() {
        positions.push(relaxed_ik.vars.goal_positions[i].x);
        positions.push(relaxed_ik.vars.goal_positions[i].y);
        positions.push(relaxed_ik.vars.goal_positions[i].z);
    }
    let ptr = positions.as_ptr();
    let len = positions.len();
    std::mem::forget(positions);
    Opt {data: ptr, length: len as c_int}
}

// This is mainly for backward compatibility
#[no_mangle]
pub unsafe extern "C" fn solve(ptr: *mut RelaxedIK, pos_goals: *const c_double, pos_length: c_int, 
    quat_goals: *const c_double, quat_length: c_int,
    tolerance: *const c_double, tolerance_length: c_int) -> Opt {

    let relaxed_ik = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };
    
    assert!(!pos_goals.is_null(), "Null pointer for pos goals!");
    assert!(!quat_goals.is_null(), "Null pointer for quat goals!");
    assert!(!tolerance.is_null(), "Null pointer for tolerance!"); 

    let pos_slice: &[c_double] = std::slice::from_raw_parts(pos_goals, pos_length as usize);
    let quat_slice: &[c_double] = std::slice::from_raw_parts(quat_goals, quat_length as usize);
    let tolerance_slice: &[c_double] = std::slice::from_raw_parts(tolerance, tolerance_length as usize);

    let pos_vec = pos_slice.to_vec();
    let quat_vec = quat_slice.to_vec();
    let tolerance_vec = tolerance_slice.to_vec();

    let ja = solve_position_helper(relaxed_ik, pos_vec, quat_vec, tolerance_vec);

    let ptr = ja.as_ptr();
    let len = ja.len();
    std::mem::forget(ja);
    Opt {data: ptr, length: len as c_int}
}

fn solve_position_helper(relaxed_ik: &mut RelaxedIK, pos_goals: Vec<f64>, quat_goals: Vec<f64>,
                tolerance: Vec<f64>) -> Vec<f64> {

    for i in 0..relaxed_ik.vars.robot.num_chains  {
        relaxed_ik.vars.goal_positions[i] = Vector3::new(pos_goals[3*i], pos_goals[3*i+1], pos_goals[3*i+2]);
        let tmp_q = Quaternion::new(quat_goals[4*i+3], quat_goals[4*i], quat_goals[4*i+1], quat_goals[4*i+2]);
        relaxed_ik.vars.goal_quats[i] =  UnitQuaternion::from_quaternion(tmp_q);
        relaxed_ik.vars.tolerances[i] = Vector6::new( tolerance[6*i], tolerance[6*i+1], tolerance[6*i+2],
            tolerance[6*i+3], tolerance[6*i+4], tolerance[6*i+5])
    }
                    
    let x = relaxed_ik.solve();
    return x;
}

fn solve_velocity_helper(relaxed_ik: &mut RelaxedIK, pos_vels: Vec<f64>, rot_vels: Vec<f64>,
    tolerance: Vec<f64>) -> Vec<f64> {

    for i in 0..relaxed_ik.vars.robot.num_chains  {
        relaxed_ik.vars.goal_positions[i] += Vector3::new(pos_vels[3*i], pos_vels[3*i+1], pos_vels[3*i+2]);
        let axisangle = Vector3::new(rot_vels[3*i], rot_vels[3*i+1], rot_vels[3*i+2]);
        let tmp_q = UnitQuaternion::from_scaled_axis(axisangle);
        let org_q = relaxed_ik.vars.goal_quats[i].clone();
        relaxed_ik.vars.goal_quats[i] =  tmp_q * org_q;
        relaxed_ik.vars.tolerances[i] = Vector6::new( tolerance[3*i], tolerance[3*i+1], tolerance[3*i+2],
            tolerance[3*i+3], tolerance[3*i+4], tolerance[3*i+5])
    }

    let x = relaxed_ik.solve();


    let frames = relaxed_ik.vars.robot.get_frames_immutable(&x);
    let last = frames[0].0.len() - 1 ;
    let ee_pos = frames[0].0[last];
    let goal = relaxed_ik.vars.goal_positions[0];
    let dist = (ee_pos - goal).norm();

    return x;
}

#[no_mangle]
pub unsafe extern "C" fn hiro_solve_velocity(ptr: *mut RelaxedIK, pos_vels: *const c_double, pos_length: c_int, 
    quat_goal: *const c_double, quat_length: c_int,
    tolerance: *const c_double, tolerance_length: c_int,
    quat_line: *const c_double, quat_line_length: c_int,
    cone_params: *const c_double, cone_params_len: c_int,
    x_a: *const c_double, x_a_len: c_int,
    x_g: *const c_double, x_g_len: c_int,
    x_hist: *const c_double, x_hist_len: c_int,
    y_hist: *const c_double, y_hist_len: c_int,
    z_hist: *const c_double, z_hist_len: c_int) -> Opt {
    
    let relaxed_ik = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };
    
    assert!(!pos_vels.is_null(), "Null pointer for pos vels!");
    assert!(!quat_goal.is_null(), "Null pointer for quat goal!");
    assert!(!quat_line.is_null(), "Null pointer for quat line!");
    assert!(!tolerance.is_null(), "Null pointer for tolerance!"); 
    assert!(!cone_params.is_null(), "Null pointer for cone!"); 
    assert!(!x_a.is_null(), "Null pointer for x_a!"); 
    assert!(!x_g.is_null(), "Null pointer for x_g!"); 
    assert!(!x_hist.is_null(), "Null pointer for x_hist!"); 
    assert!(!y_hist.is_null(), "Null pointer for y_hist!"); 
    assert!(!z_hist.is_null(), "Null pointer for z_hist!"); 

    assert!(pos_length as usize == relaxed_ik.vars.robot.num_chains * 3 , 
        "Pos vels are expected to have {} numbers, but got {}", 
        relaxed_ik.vars.robot.num_chains * 3, pos_length);
    assert!(quat_length as usize == relaxed_ik.vars.robot.num_chains * 4,
        "Rot vels are expected to have {} numbers, but got {}", 
        relaxed_ik.vars.robot.num_chains * 3, quat_length);
    assert!(tolerance_length as usize == relaxed_ik.vars.robot.num_chains * 6, 
        "Tolerance are expected to have {} numbers, but got {}", 
        relaxed_ik.vars.robot.num_chains * 6, tolerance_length);
    assert!(quat_line_length as usize == relaxed_ik.vars.robot.num_chains * 3, 
        "Quaternion line are expected to have {} numbers, but got {}", 
        relaxed_ik.vars.robot.num_chains * 3, quat_line_length);
    assert!(cone_params_len as usize == relaxed_ik.vars.robot.num_chains * 4, 
        "Cone params are expected to have {} numbers, but got {}", 
        relaxed_ik.vars.robot.num_chains * 4, cone_params_len);
    assert!(x_a_len as usize == relaxed_ik.vars.robot.num_chains * 3 , 
        "x_a are expected to have {} numbers, but got {}", 
        relaxed_ik.vars.robot.num_chains * 3, x_a_len);
    assert!(x_g_len as usize == relaxed_ik.vars.robot.num_chains * 3 , 
        "x_g  are expected to have {} numbers, but got {}", 
        relaxed_ik.vars.robot.num_chains * 3, x_g_len);


    let pos_slice: &[c_double] = std::slice::from_raw_parts(pos_vels, pos_length as usize);
    let quat_slice: &[c_double] = std::slice::from_raw_parts(quat_goal, quat_length as usize);
    let tolerance_slice: &[c_double] = std::slice::from_raw_parts(tolerance, tolerance_length as usize);
    let quat_line_slice: &[c_double] = std::slice::from_raw_parts(quat_line, quat_line_length as usize);
    let cone_param_slice: &[c_double] = std::slice::from_raw_parts(cone_params, cone_params_len as usize);
    let x_a_slice: &[c_double] = std::slice::from_raw_parts(x_a, x_a_len as usize);
    let x_g_slice: &[c_double] = std::slice::from_raw_parts(x_g, x_g_len as usize);
    let x_hist_slice: &[c_double] = std::slice::from_raw_parts(x_hist, x_hist_len as usize);
    let y_hist_slice: &[c_double] = std::slice::from_raw_parts(y_hist, y_hist_len as usize);
    let z_hist_slice: &[c_double] = std::slice::from_raw_parts(z_hist, z_hist_len as usize);

    let pos_vec = pos_slice.to_vec();
    let quat_vec = quat_slice.to_vec();
    let tolerance_vec = tolerance_slice.to_vec();
    let quat_line_vec = quat_line_slice.to_vec();
    let cone_params_vec = cone_param_slice.to_vec();
    let x_a_vec = x_a_slice.to_vec();
    let x_g_vec = x_g_slice.to_vec();
    let x_hist_vec = x_hist_slice.to_vec();
    let y_hist_vec = y_hist_slice.to_vec();
    let z_hist_vec = z_hist_slice.to_vec();

    relaxed_ik.vars.radius = cone_params_vec[0];
    relaxed_ik.vars.height = cone_params_vec[1];
    relaxed_ik.vars.obj_to_center_line = cone_params_vec[2];
    relaxed_ik.vars.start_cone = cone_params_vec[3];
    relaxed_ik.vars.x_a = x_a_vec;
    relaxed_ik.vars.x_g = x_g_vec;
    relaxed_ik.vars.x_hist = x_hist_vec;
    relaxed_ik.vars.y_hist = y_hist_vec;
    relaxed_ik.vars.z_hist = z_hist_vec;

    // println!("{}\n, {}\n, {}\n, {}\n", relaxed_ik.radius, relaxed_ik.height, relaxed_ik.obj_to_center_line, relaxed_ik.start_cone);

    let ja = hiro_solve_velocity_helper(relaxed_ik, pos_vec, quat_vec, tolerance_vec, quat_line_vec);

    let ptr = ja.as_ptr();
    let len = ja.len();
    std::mem::forget(ja);
    Opt {data: ptr, length: len as c_int}
}


fn hiro_solve_velocity_helper(relaxed_ik: &mut RelaxedIK, pos_vels: Vec<f64>, quat_goal: Vec<f64>,
    tolerance: Vec<f64>, quat_line: Vec<f64>) -> Vec<f64> {

    for i in 0..relaxed_ik.vars.robot.num_chains  {
        relaxed_ik.vars.goal_positions[i] += Vector3::new(pos_vels[3*i], pos_vels[3*i+1], pos_vels[3*i+2]);
        let tmp_quat = Quaternion::new(quat_goal[3*i], quat_goal[3*i+1], quat_goal[3*i+2], quat_goal[3*i+3]);
        let tmp_quat_norm = tmp_quat.normalize();
        let quaternion1 = UnitQuaternion::from_quaternion(tmp_quat_norm);

        let axis = Unit::new_normalize(Vector3::new(-quat_line[3*i], -quat_line[3*i+1], -quat_line[3*i+2]));
        let rot = UnitQuaternion::from_axis_angle(&axis, 0.785398);
        let corrected_quat = rot * quaternion1;
        // relaxed_ik.vars.goal_quats[i] =  quaternion1;
        relaxed_ik.vars.goal_quats[i] =  corrected_quat;
        relaxed_ik.vars.tolerances[i] = Vector6::new( tolerance[3*i], tolerance[3*i+1], tolerance[3*i+2],
            tolerance[3*i+3], tolerance[3*i+4], tolerance[3*i+5])
    }
    let x = relaxed_ik.solve();


    let frames = relaxed_ik.vars.robot.get_frames_immutable(&x);
    let last = frames[0].0.len() - 1 ;
    let ee_pos = frames[0].0[last];
    let goal = relaxed_ik.vars.goal_positions[0];
    let dist = (ee_pos - goal).norm();

    return x;
}
