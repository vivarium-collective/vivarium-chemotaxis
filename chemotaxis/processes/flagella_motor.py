'''
==========================
Flagella Motor Processes
==========================
'''

import os
import copy
import random
import math
import uuid

import numpy as np

from vivarium.core.process import Process
from vivarium.core.composition import (
    simulate_process_in_experiment,
    plot_simulation_output,
    PROCESS_OUT_DIR,
)
from vivarium.core.emitter import timeseries_from_data

# plots
from chemotaxis.plots.flagella_activity import plot_activity


NAME = 'flagella_motor'


class FlagellaMotor(Process):
    '''
    Model of flagellarmotor activity
    '''

    name = NAME
    initial_pmf = 170  # PMF ~170mV at pH 7, ~140mV at pH 7.7 (Berg H, E. coli in motion, 2004, pg 113)
    defaults = {
        'time_step': 0.01,

        # Vladimirov parameters
        # 'k_A': 5.0,  #
        'k_y': 100.0,  # 1/uM/s
        'k_z': 30.0,  # / CheZ,
        'gamma_Y': 0.1,  # rate constant
        'k_s': 0.45,  # scaling coefficient
        'adapt_precision': 8,  # scales CheY_P to cluster activity

        # motor
        'mb_0': 0.65,  # steady state motor bias (Cluzel et al 2000)
        'n_motors': 5,
        'cw_to_ccw': 0.83,  # 1/s (Block1983) motor bias, assumed to be constant
        'cw_to_ccw_leak': 0.25,  # rate of spontaneous transition to tumble

        # parameters for multibody physics
        'tumble_jitter': 120.0,

        # rotational state of individual flagella
        # parameters from Sneddon, Pontius, and Emonet (2012)
        'omega': 1.3,  # (1/s) characteristic motor switch time
        'g_0': 40,  # (k_B/T) free energy barrier for CCW-->CW
        'g_1': 40,  # (k_B/T) free energy barrier for CW-->CCW
        'K_D': 3.06,  # binding constant of Chey-P to base of the motor

        # motile force parameters
        'flagellum_thrust': 25,  # (pN) (Berg H, E. coli in motion, 2004, pg 113)
        'tumble_jitter': 120.0,
        'tumble_scaling': 1.4 / initial_pmf,
        'run_scaling': 1.4 / initial_pmf,

        # initial state
        'initial_state': {
            'internal': {
                # response regulator proteins
                'CheY': 2.59,
                'CheY_P': 2.59,  # (uM) mean concentration of CheY-P
                'CheZ': 1.0,  # (uM) #phosphatase 100 uM = 0.1 mM
                'CheA': 1.0,  # (uM) #100 uM = 0.1 mM
                # sensor activity
                'chemoreceptor_activity': 1/3,
                # motor activity
                'ccw_motor_bias': 0.5,
                'cw_bias': 0.5,
                'motile_state': 1,  # 1 for tumble, 0 for run
            },
            'membrane': {
                'PMF': initial_pmf,
            },
            'boundary': {
                'thrust': 0,
                'torque': 0,
            }
        }
    }

    def __init__(self, parameters=None):
        super(FlagellaMotor, self).__init__(parameters)

    def ports_schema(self):
        ports = [
            'internal_counts',
            'flagella',
            'internal',
            'boundary',
            'membrane',
        ]
        schema = {port: {} for port in ports}

        # internal_counts of flagella (n_flagella)
        schema['internal_counts']['flagella'] = {
            '_value': self.parameters['n_flagella'],
            '_default': self.parameters['n_flagella'],
            '_emit': True}

        # flagella
        schema['flagella'] = {
            '_divider': 'split_dict',
            '*': {
                '_default': 1,
                '_updater': 'set',
                '_emit': True}}

        # internal
        state_emit = [
            'chemoreceptor_activity',
            'ccw_motor_bias',
            'cw_bias',
            'motile_state',
            'CheA',
            'CheY_P',
            'CheY',
        ]
        state_set_updater = [
                'ccw_motor_bias',
                'cw_bias',
                'motile_state',
                'CheA',
                'CheY_P',
                'CheY',
        ]
        for state, default in self.parameters['initial_state']['internal'].items():
            schema['internal'][state] = {'_default': default}
            if state in state_emit:
                schema['internal'][state].update({
                    '_emit': True})
            if state in state_set_updater:
                schema['internal'][state].update({
                    '_updater': 'set'})

        # boundary (thrust and torque)
        for state, default in self.parameters['initial_state']['boundary'].items():
            schema['boundary'][state] = {
                '_default': default,
                '_emit': True,
                '_updater': 'set'}

        # membrane
        for state in ['PMF', 'protons_flux_accumulated']:
            schema['membrane'][state] = {
                '_default': self.parameters['initial_state']['membrane'].get(state, 0.0)}

        return schema

    def next_update(self, timestep, states):

        # get flagella subcompartments and current flagella counts
        flagella = states['flagella']
        n_flagella = states['internal_counts']['flagella']

        # proton motive force
        PMF = states['membrane']['PMF']

        # get internal states
        internal = states['internal']
        P_on = internal['chemoreceptor_activity']
        CheY_0 = internal['CheY']
        CheY_P_0 = internal['CheY_P']

        # parameters
        adapt_precision = self.parameters['adapt_precision']
        k_y = self.parameters['k_y']
        k_s = self.parameters['k_s']
        k_z = self.parameters['k_z']
        gamma_Y = self.parameters['gamma_Y']
        mb_0 = self.parameters['mb_0']
        cw_to_ccw = self.parameters['cw_to_ccw']

        ## Kinase activity
        # relative steady-state concentration of phosphorylated CheY.
        new_CheY_P = adapt_precision * k_y * k_s * P_on / (k_y * k_s * P_on + k_z + gamma_Y)  # CheZ cancels out of k_z
        dCheY_P = new_CheY_P - CheY_P_0
        CheY_P = max(new_CheY_P, 0.0)  # keep value positive
        CheY = max(CheY_0 - dCheY_P, 0.0)  # keep value positive

        ## Motor switching
        # CCW corresponds to run. CW corresponds to tumble
        ccw_motor_bias = mb_0 / (CheY_P * (1 - mb_0) + mb_0)  # (1/s)
        cw_bias = cw_to_ccw * (1 / ccw_motor_bias - 1)  # (1/s)
        # don't let cw_bias get under leak value
        if cw_bias < self.parameters['cw_to_ccw_leak']:
            cw_bias = self.parameters['cw_to_ccw_leak']

        ## update flagella
        # update number of flagella
        flagella_update = {}
        new_flagella = int(n_flagella) - len(flagella)
        if new_flagella < 0:
            # remove flagella
            flagella_update['_delete'] = []
            remove = random.sample(list(flagella.keys()), abs(new_flagella))
            for flagella_id in remove:
                flagella_update['_delete'].append((flagella_id,))

        elif new_flagella > 0:
            # add flagella
            flagella_update['_add'] = []
            for index in range(new_flagella):
                flagella_id = str(uuid.uuid1())
                flagella_update['_add'].append({
                    'path': (flagella_id, ),
                    'state': random.choice([-1, 1])})

        # update flagella states
        for flagella_id, motor_state in flagella.items():
            new_motor_state = self.update_flagellum(motor_state, cw_bias, CheY_P, timestep)
            flagella_update.update({flagella_id: new_motor_state})

        ## get cell motile state.
        # if any flagella is rotating CW, the cell tumbles.
        # flagella motor state: -1 for CCW, 1 for CW
        # motile state: -1 for run, 1 for tumble, 0 for no state
        if any(state == 1 for state in flagella_update.values()):
            motile_state = 1
            [thrust, torque] = self.tumble(n_flagella, PMF)
        elif len(flagella_update) > 0:
            motile_state = -1
            [thrust, torque] = self.run(n_flagella, PMF)
        else:
            motile_state = 0
            thrust = 0
            torque = 0

        return {
            'flagella': flagella_update,
            'internal': {
                'ccw_motor_bias': ccw_motor_bias,
                'cw_bias': cw_bias,
                'motile_state': motile_state,
                'CheY_P': CheY_P,
                'CheY': CheY
            },
            'boundary': {
                'thrust': thrust,
                'torque': torque
            }}


    def update_flagellum(self, motor_state, cw_bias, CheY_P, timestep):
        '''
        Rotational state of an individual flagellum from:
            Sneddon, M. W., Pontius, W., & Emonet, T. (2012).
            Stochastic coordination of multiple actuators reduces
            latency and improves chemotactic response in bacteria.

        # TODO -- normal, semi, curly states from Sneddon
        '''
        g_0 = self.parameters['g_0']  # (k_B/T) free energy barrier for CCW-->CW
        g_1 = self.parameters['g_1']  # (k_B/T) free energy barrier for CW-->CCW
        K_D = self.parameters['K_D']  # binding constant of CheY-P to base of the motor
        omega = self.parameters['omega']  # (1/s) characteristic motor switch time

        # free energy barrier
        delta_g = g_0 / 4 - g_1 / 2 * (CheY_P / (CheY_P + K_D))

        # switching frequency
        CW_to_CCW = omega * math.exp(delta_g)
        CCW_to_CW = omega * math.exp(-delta_g)
        # switch_freq = CCW_to_CW * (1 - cw_bias) + CW_to_CCW * cw_bias

        # flagella motor state: -1 for CCW, 1 for CW
        if motor_state == -1:
            prob_switch = CCW_to_CW * timestep
            if np.random.random(1)[0] <= prob_switch:
                new_motor_state = 1
            else:
                new_motor_state = -1

        elif motor_state == 1:
            prob_switch = CW_to_CCW * timestep
            if np.random.random(1)[0] <= prob_switch:
                new_motor_state = -1
            else:
                new_motor_state = 1

        return new_motor_state

    def tumble(self, n_flagella, PMF):
        # thrust scales with lg(n_flagella) because only the thickness of the bundle is affected
        thrust = self.parameters['tumble_scaling'] * PMF * self.parameters['flagellum_thrust'] * math.log(n_flagella + 1)
        torque = random.normalvariate(0, self.parameters['tumble_jitter'])
        return [thrust, torque]

    def run(self, n_flagella, PMF):
        # thrust scales with lg(n_flagella) because only the thickness of the bundle is affected
        thrust = self.parameters['run_scaling'] * PMF * self.parameters['flagellum_thrust'] * math.log(n_flagella + 1)
        torque = 0.0
        return [thrust, torque]


# testing functions
def get_chemoreceptor_timeline(
        total_time=2,
        time_step=0.01,
        rate=1.0,
        initial_value=1.0/3.0
):
    val = copy.copy(initial_value)
    timeline = [(0, {('internal', 'chemoreceptor_activity'): initial_value})]
    t = 0
    while t < total_time:
        val += random.choice((-1, 1)) * rate * time_step
        if val < 0:
            val = 0
        if val > 1:
            val = 1
        timeline.append((t, {('internal', 'chemoreceptor_activity'): val}))
        t += time_step
    return timeline


default_timeline = [(10, {})]
default_params = {'n_flagella': 3}
def test_flagella_motor(
        timeline=default_timeline,
        time_step=0.01,
        parameters=default_params
):
    motor = FlagellaMotor(parameters)
    settings = {
        'return_raw_data': True,
        'timeline': {
            'timeline': timeline,
            'time_step': time_step}}
    return simulate_process_in_experiment(motor, settings)


def run_variable_flagella(out_dir='out'):
    time_step = 0.01
    # make timeline with both chemoreceptor variation and flagella counts
    timeline = get_chemoreceptor_timeline(
        total_time=3,
        time_step=time_step,
        rate=2.0,
    )
    timeline_flagella = [
        (0.0, {('internal_counts', 'flagella'): 0}),
        (0.5, {('internal_counts', 'flagella'): 1}),
        (1.0, {('internal_counts', 'flagella'): 2}),
        (1.5, {('internal_counts', 'flagella'): 3}),
        (2.0, {('internal_counts', 'flagella'): 4}),
        (2.5, {('internal_counts', 'flagella'): 5}),
    ]
    timeline.extend(timeline_flagella)

    # run simulation
    data = test_flagella_motor(
        timeline=timeline,
        time_step=time_step,
    )

    # plot
    plot_settings = {}
    timeseries = timeseries_from_data(data)
    plot_simulation_output(timeseries, plot_settings, out_dir)
    plot_activity(data, plot_settings, out_dir)


if __name__ == '__main__':
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    run_variable_flagella(out_dir)
