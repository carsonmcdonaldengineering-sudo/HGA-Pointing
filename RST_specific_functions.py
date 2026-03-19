from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize

import rotation_functions as rot


# HGA defaults pulled from Cycle 4 STOP Observatory Thermal Model
dish_pointing_0_0 = np.array([-0.17384746, -0.00824553, 0.98473807])
y_gimbal_rotation_axis = np.array([-0.00423894, 0.99996195, 0.00762465])
x_gimbal_rotation_axis = np.array([0.98476347, 0.00284872, 0.1738758])
HGA_initial_configuration = [dish_pointing_0_0, y_gimbal_rotation_axis, x_gimbal_rotation_axis]

# defaults for observatory rotations in FOR
yaw_obs_rotation_axis = np.array([0, 0, 1])
pitch_obs_rotation_axis = np.array([0, 1, 0])
roll_obs_rotation_axis = np.array([1, 0, 0])
sun_angle_0_0 = np.array([0, 0, 1])
obs_initial = [sun_angle_0_0, yaw_obs_rotation_axis, pitch_obs_rotation_axis, roll_obs_rotation_axis]

DEFAULT_GIMBAL_BOUNDS = [(-87, 75), (-54, 54)]


@dataclass(frozen=True)
class ObservatoryAttitude:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    @classmethod
    def from_input(cls, attitude: Sequence[float] | 'ObservatoryAttitude') -> 'ObservatoryAttitude':
        if isinstance(attitude, cls):
            return attitude
        if len(attitude) == 2:
            pitch, roll = attitude
            return cls(yaw=0.0, pitch=pitch, roll=roll)
        if len(attitude) == 3:
            yaw, pitch, roll = attitude
            return cls(yaw=yaw, pitch=pitch, roll=roll)
        raise ValueError('Attitude must be an ObservatoryAttitude or a sequence of length 2 or 3.')

    def as_tuple(self):
        return self.yaw, self.pitch, self.roll

    def label(self):
        return f'Yaw {self.yaw} Pitch {self.pitch} Roll {self.roll}'


@dataclass(frozen=True)
class GimbalAngles:
    y_track: float
    x_track: float

    def as_tuple(self):
        return self.y_track, self.x_track




@dataclass(frozen=True)
class ThermalDesktopExports:
    x_track: str
    y_track: str


@dataclass
class SolveResult:
    attitude: ObservatoryAttitude
    gimbal_angles: GimbalAngles
    pointing_error: float
    success: bool
    iterations: int


class RomanHGAPointingModel:
    def __init__(self, hga_initial_config=None, observatory_axes=None, gimbal_bounds=None):
        self.hga_initial_config = hga_initial_config or HGA_initial_configuration
        self.observatory_axes = observatory_axes or obs_initial[1:]
        self.gimbal_bounds = gimbal_bounds or DEFAULT_GIMBAL_BOUNDS

    def rotate_hga(self, gimbal_angles, hga_initial_config=None):
        y_track, x_track = GimbalAngles(*gimbal_angles).as_tuple()
        config = hga_initial_config or self.hga_initial_config
        return rot.rotate_two_vectors(config[0], y_track, x_track, config[1], config[2])

    def rotate_hga_coordinates_within_observatory(self, attitude, hga_points=None):
        attitude = ObservatoryAttitude.from_input(attitude)
        yaw, pitch, roll = attitude.as_tuple()
        hga_points = hga_points or self.hga_initial_config

        sun_vector = rot.rotate_three_vectors(
            obs_initial[0],
            yaw,
            pitch,
            roll,
            *self.observatory_axes,
        )

        rotated_hga = [
            rot.rotate_three_vectors(point, yaw, pitch, roll, *self.observatory_axes)
            for point in hga_points
        ]
        return rotated_hga, sun_vector

    def define_target(self, gimbal_angles, attitude=(0, 0, 0), verbose=True):
        attitude = ObservatoryAttitude.from_input(attitude)
        gimbal_angles = GimbalAngles(*gimbal_angles)
        rotated_hga, sun_vector = self.rotate_hga_coordinates_within_observatory(attitude)
        target = self.rotate_hga(gimbal_angles.as_tuple(), rotated_hga)

        if verbose:
            print(f'Target for {attitude.label()}')
            print(f'With HGA y_track = {gimbal_angles.y_track} and x_track = {gimbal_angles.x_track}')
            print(target)
            print()
            print(f'Angle between HGA pointing & sun = {rot.angle_between_vectors(sun_vector, target)}')
            print()

        return target

    def pointing_error_function(self, attitude, target_vector):
        rotated_hga, _ = self.rotate_hga_coordinates_within_observatory(attitude)

        def distance_to_target(gimbal_inputs):
            pointing = self.rotate_hga(gimbal_inputs, rotated_hga)
            return np.linalg.norm(pointing - target_vector)

        return distance_to_target

    def solve_gimbal_inputs(self, attitude, target_vector, initial_guess=(0, 0)):
        attitude = ObservatoryAttitude.from_input(attitude)
        objective = self.pointing_error_function(attitude, target_vector)
        result = minimize(objective, initial_guess, bounds=self.gimbal_bounds)
        return SolveResult(
            attitude=attitude,
            gimbal_angles=GimbalAngles(*result.x),
            pointing_error=float(result.fun),
            success=bool(result.success),
            iterations=int(getattr(result, 'nit', 0)),
        )

    def solve_across_attitudes(self, target_vector, attitudes: Iterable[Sequence[float]], initial_guess=(0, 0)):
        return [self.solve_gimbal_inputs(attitude, target_vector, initial_guess) for attitude in attitudes]

    def format_thermal_desktop_table(
        self,
        results,
        axis='x',
        pitch_symbol='STOP_obs_pitch',
        roll_symbol='STOP_obs_roll',
        value_format='.9g',
    ):
        axis_lookup = {
            'x': 'x_track',
            'x_track': 'x_track',
            'y': 'y_track',
            'y_track': 'y_track',
        }
        axis_name = axis_lookup.get(axis)
        if axis_name is None:
            raise ValueError("axis must be one of 'x', 'x_track', 'y', or 'y_track'.")

        lines = []
        for result in results:
            attitude = ObservatoryAttitude.from_input(result.attitude)
            if attitude.yaw != 0:
                raise ValueError('Thermal Desktop export currently supports only yaw = 0 attitude tables.')

            value = getattr(result.gimbal_angles, axis_name)
            lines.append(
                f'(({pitch_symbol}== {attitude.pitch:g}) && ({roll_symbol}== {attitude.roll:g})) ?    {format(value, value_format)}:'
            )

        lines.append('0')
        return '\n'.join(lines)

    def format_thermal_desktop_gimbal_exports(
        self,
        results,
        pitch_symbol='STOP_obs_pitch',
        roll_symbol='STOP_obs_roll',
        value_format='.9g',
    ):
        return ThermalDesktopExports(
            x_track=self.format_thermal_desktop_table(
                results,
                axis='x',
                pitch_symbol=pitch_symbol,
                roll_symbol=roll_symbol,
                value_format=value_format,
            ),
            y_track=self.format_thermal_desktop_table(
                results,
                axis='y',
                pitch_symbol=pitch_symbol,
                roll_symbol=roll_symbol,
                value_format=value_format,
            ),
        )


def rotate_HGA(HGA_inputs, HGA_initial_config=HGA_initial_configuration):
    return RomanHGAPointingModel(hga_initial_config=HGA_initial_config).rotate_hga(HGA_inputs, HGA_initial_config)


def rotate_HGA_coordinates_within_OBS(obs_FOR_attitude, HGA_point=HGA_initial_configuration):
    return RomanHGAPointingModel(hga_initial_config=HGA_point).rotate_hga_coordinates_within_observatory(obs_FOR_attitude, HGA_point)


def define_target(HGA_inputs, obs_FOR_attitude=(0, 0, 0), verbose=True):
    return RomanHGAPointingModel().define_target(HGA_inputs, obs_FOR_attitude, verbose=verbose)


__all__ = [
    'DEFAULT_GIMBAL_BOUNDS',
    'GimbalAngles',
    'ObservatoryAttitude',
    'RomanHGAPointingModel',
    'SolveResult',
    'ThermalDesktopExports',
    'define_target',
    'rotate_HGA',
    'rotate_HGA_coordinates_within_OBS',
]
