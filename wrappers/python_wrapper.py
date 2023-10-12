#! /usr/bin/env python3

import ctypes
import os

class Opt(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_double)), ("length", ctypes.c_int)]

class RelaxedIKS(ctypes.Structure):
    pass

dir_path = os.path.dirname(os.path.realpath(__file__))
lib = ctypes.cdll.LoadLibrary(dir_path + '/../target/debug/librelaxed_ik_lib.so')

lib.relaxed_ik_new.restype = ctypes.POINTER(RelaxedIKS)
lib.solve.argtypes = [ctypes.POINTER(RelaxedIKS), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
lib.solve.restype = Opt
lib.solve_position.argtypes = [ctypes.POINTER(RelaxedIKS), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
lib.solve_position.restype = Opt
lib.solve_velocity.argtypes = [ctypes.POINTER(RelaxedIKS), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
lib.solve_velocity.restype = Opt
lib.hiro_solve_velocity.argtypes = [ctypes.POINTER(RelaxedIKS), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int, 
                                    ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
                                    ctypes.POINTER(ctypes.c_double), ctypes.c_int,
                                    ctypes.POINTER(ctypes.c_double), ctypes.c_int,
                                    ctypes.POINTER(ctypes.c_double), ctypes.c_int]
lib.hiro_solve_velocity.restype = Opt

class RelaxedIKRust:
    def __init__(self, setting_file_path = None):
        '''
        setting_file_path (string): path to the setting file
                                    if no path is given, the default setting file will be used
                                    /configs/settings.yaml
        '''
        if setting_file_path is None:
            self.obj = lib.relaxed_ik_new(ctypes.c_char_p())
        else:
            self.obj = lib.relaxed_ik_new(ctypes.c_char_p(setting_file_path.encode('utf-8')))
    
    def __exit__(self, exc_type, exc_value, traceback):
        lib.relaxed_ik_free(self.obj)
    
    def solve_position(self, positions, orientations, tolerances):
        '''
        Assuming the robot has N end-effectors
        positions (1D array with length as 3*N): list of end-effector positions
        orientations (1D array with length as 4*N): list of end-effector orientations (in quaternion xyzw format)
        tolerances (1D array with length as 6*N): list of tolerances for each end-effector (x, y, z, rx, ry, rz)
        '''
        pos_arr = (ctypes.c_double * len(positions))()
        quat_arr = (ctypes.c_double * len(orientations))()
        tole_arr = (ctypes.c_double * len(tolerances))()
        for i in range(len(positions)):
            pos_arr[i] = positions[i]
        for i in range(len(orientations)):
            quat_arr[i] = orientations[i]
        for i in range(len(tolerances)):
            tole_arr[i] = tolerances[i]
        xopt = lib.solve_position(self.obj, pos_arr, len(pos_arr), quat_arr, len(quat_arr), tole_arr, len(tole_arr))
        return xopt.data[:xopt.length]
    
    def solve_velocity(self, linear_velocities, angular_velocities, tolerances):
        '''
        Assuming the robot has N end-effectors
        linear_velocities (1D array with length as 3*N): list of end-effector linear velocities
        angular_velocities (1D array with length as 4*N): list of end-effector angular velocities
        tolerances (1D array with length as 6*N): list of tolerances for each end-effector (x, y, z, rx, ry, rz)
        '''
        
        linear_arr = (ctypes.c_double * len(linear_velocities))()
        angular_arr = (ctypes.c_double * len(angular_velocities))()
        tole_arr = (ctypes.c_double * len(tolerances))()
        for i in range(len(linear_velocities)):
            linear_arr[i] = linear_velocities[i]
        for i in range(len(angular_velocities)):
            angular_arr[i] = angular_velocities[i]
        for i in range(len(tolerances)):
            tole_arr[i] = tolerances[i]
        xopt = lib.solve_velocity(self.obj, linear_arr, len(linear_arr), angular_arr, len(angular_arr), tole_arr, len(tole_arr))
        return xopt.data[:xopt.length]
    
    def hiro_solve_velocity(self, linear_velocities, quat_goal, tolerances, quat_line, cone_params, x_a, x_g, x_hist, y_hist, z_hist):
        '''
        Assuming the robot has N end-effectors
        linear_velocities (1D array with length as 3*N): list of end-effector linear velocities
        angular_velocities (1D array with length as 4*N): list of end-effector angular velocities
        tolerances (1D array with length as 6*N): list of tolerances for each end-effector (x, y, z, rx, ry, rz)
        '''
        
        linear_arr = (ctypes.c_double * len(linear_velocities))()
        angular_arr = (ctypes.c_double * len(quat_goal))()
        tole_arr = (ctypes.c_double * len(tolerances))()
        quat_line_arr = (ctypes.c_double * len(quat_line))()
        cone_params_arr = (ctypes.c_double * len(cone_params))()
        x_a_arr = (ctypes.c_double * len(x_a))()
        x_g_arr = (ctypes.c_double * len(x_g))()
        x_hist_arr = (ctypes.c_double * len(x_hist))()
        y_hist_arr = (ctypes.c_double * len(y_hist))()
        z_hist_arr = (ctypes.c_double * len(z_hist))()
        
        for i in range(len(linear_velocities)):
            linear_arr[i] = linear_velocities[i]
        for i in range(len(quat_goal)):
            angular_arr[i] = quat_goal[i]
        for i in range(len(tolerances)):
            tole_arr[i] = tolerances[i]
        for i in range(len(quat_line)):
            quat_line_arr[i] = quat_line[i]
        for i in range(len(cone_params)):
            cone_params_arr[i] = cone_params[i]
        for i in range(len(x_a)):
            x_a_arr[i] = x_a[i]
        for i in range(len(x_g)):
            x_g_arr[i] = x_g[i]

        for i in range(len(x_hist_arr)):
            x_hist_arr[i] = x_hist[i]
        for i in range(len(y_hist_arr)):
            y_hist_arr[i] = y_hist[i]
        for i in range(len(z_hist_arr)):
            z_hist_arr[i] = z_hist[i]
        
        xopt = lib.hiro_solve_velocity(self.obj, linear_arr, len(linear_arr), angular_arr, len(angular_arr), tole_arr, len(tole_arr), quat_line_arr, len(quat_line_arr),
                                       cone_params_arr, len(cone_params_arr), x_a_arr, len(x_a_arr), x_g_arr, len(x_g_arr),
                                       x_hist_arr, len(x_hist_arr), y_hist_arr, len(y_hist_arr), z_hist_arr, len(z_hist_arr))
        return xopt.data[:xopt.length]

if __name__ == '__main__':
    pass
