import os
import math

import numpy as np

import matplotlib.pyplot as plt


PI = math.pi


def plot_motility(timeseries, out_dir='out', filename='motility_analysis'):

    expected_velocity = 14.2  # um/s (Berg)
    expected_angle_between_runs = 68 # degrees (Berg)

    # time of motor behavior without chemotaxis
    expected_run_duration = 0.42  # s (Berg)
    expected_tumble_duration = 0.14  # s (Berg)

    times = timeseries['time']
    agents = timeseries['agents']

    motility_analysis = {
        agent_id: {
            'velocity': [],
            'angular_velocity': [],
            'angle_between_runs': [],
            'angle': [],
            'thrust': [],
            'torque': [],
            'run_duration': [],
            'tumble_duration': [],
            'run_time': [],
            'tumble_time': [],
        }
        for agent_id in list(agents.keys())}

    for agent_id, agent_data in agents.items():

        boundary_data = agent_data['boundary']
        cell_data = agent_data['internal']
        previous_time = times[0]
        previous_angle = boundary_data['angle'][0]
        previous_location = boundary_data['location'][0]
        previous_run_angle = boundary_data['angle'][0]
        previous_motile_state = cell_data['motile_state'][0]  # 1 for tumble, -1 for run
        run_duration = 0.0
        tumble_duration = 0.0
        dt = 0.0

        # go through each time point for this agent
        for time_idx, time in enumerate(times):
            motile_state = cell_data['motile_state'][time_idx]
            angle = boundary_data['angle'][time_idx]
            location = boundary_data['location'][time_idx]
            thrust = boundary_data['thrust'][time_idx]
            torque = boundary_data['torque'][time_idx]

            # get velocity
            if time != times[0]:
                dt = time - previous_time
                distance = (
                    (location[0] - previous_location[0]) ** 2 +
                    (location[1] - previous_location[1]) ** 2
                        ) ** 0.5
                velocity = distance / dt  # um/sec

                angle_change = ((angle - previous_angle) / PI * 180) % 360
                if angle_change > 180:
                    angle_change = 360 - angle_change
                angular_velocity = angle_change/ dt
            else:
                velocity = 0.0
                angular_velocity = 0.0

            # get angle change between runs
            angle_between_runs = None
            if motile_state == -1:  # run
                if previous_motile_state == 1:
                    angle_between_runs = angle - previous_run_angle
                previous_run_angle = angle

            # get run and tumble durations
            if motile_state == -1:  # run
                if previous_motile_state == 1:
                    # the run just started -- save the previous tumble time and reset to 0
                    motility_analysis[agent_id]['tumble_duration'].append(tumble_duration)
                    motility_analysis[agent_id]['tumble_time'].append(time)
                    tumble_duration = 0
                elif previous_motile_state == -1:
                    # the run is continuing
                    run_duration += dt
            elif motile_state == 1:
                if previous_motile_state == -1:
                    # the tumble just started -- save the previous run time and reset to 0
                    motility_analysis[agent_id]['run_duration'].append(run_duration)
                    motility_analysis[agent_id]['run_time'].append(time)
                    run_duration = 0
                elif previous_motile_state == 1:
                    # the tumble is continuing
                    tumble_duration += dt

            # save data
            motility_analysis[agent_id]['velocity'].append(velocity)
            motility_analysis[agent_id]['angular_velocity'].append(angular_velocity)
            motility_analysis[agent_id]['angle'].append(angle)
            motility_analysis[agent_id]['thrust'].append(thrust)
            motility_analysis[agent_id]['torque'].append(torque)
            motility_analysis[agent_id]['angle_between_runs'].append(angle_between_runs)

            # save previous location and time
            previous_location = location
            previous_angle = angle
            previous_time = time
            previous_motile_state = motile_state

    # plot results
    cols = 1
    rows = 7
    fig = plt.figure(figsize=(6 * cols, 1.2 * rows))
    plt.rcParams.update({'font.size': 12})

    # plot velocity
    ax1 = plt.subplot(rows, cols, 1)
    for agent_id, analysis in motility_analysis.items():
        velocity = analysis['velocity']
        mean_velocity = np.mean(velocity)
        ax1.plot(times, velocity, label=agent_id)
        ax1.axhline(y=mean_velocity, linestyle='dashed', label='mean_' + agent_id)
    ax1.axhline(y=expected_velocity, color='r', linestyle='dashed', label='expected mean')
    ax1.set_ylabel(u'velocity \n (\u03bcm/sec)')
    ax1.set_xlabel('time')
    # ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # plot angular velocity
    ax2 = plt.subplot(rows, cols, 2)
    for agent_id, analysis in motility_analysis.items():
        angular_velocity = analysis['angular_velocity']
        ax2.plot(times, angular_velocity, label=agent_id)
    ax2.set_ylabel(u'angular velocity \n (degrees/sec)')
    ax2.set_xlabel('time')
    # ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # plot thrust
    ax3 = plt.subplot(rows, cols, 3)
    for agent_id, analysis in motility_analysis.items():
        thrust = analysis['thrust']
        ax3.plot(times, thrust, label=agent_id)
    ax3.set_ylabel('thrust')

    # plot torque
    ax4 = plt.subplot(rows, cols, 4)
    for agent_id, analysis in motility_analysis.items():
        torque = analysis['torque']
        ax4.plot(times, torque, label=agent_id)
    ax4.set_ylabel('torque')

    # plot angles between runs
    ax5 = plt.subplot(rows, cols, 5)
    for agent_id, analysis in motility_analysis.items():
        # convert to degrees
        angle_between_runs = [
            (angle / PI * 180) % 360 if angle is not None else None
            for angle in analysis['angle_between_runs']]
        # pair with time
        run_angle_points = [
            [t, angle] if angle < 180 else [t, 360 - angle]
            for t, angle in dict(zip(times, angle_between_runs)).items()
            if angle is not None]

        plot_times = [point[0] for point in run_angle_points]
        plot_angles = [point[1] for point in run_angle_points]
        mean_angle_change = np.mean(plot_angles)
        ax5.scatter(plot_times, plot_angles, label=agent_id)
        ax5.axhline(y=mean_angle_change, linestyle='dashed') #, label='mean_' + agent_id)
    ax5.set_ylabel(u'degrees \n between runs')
    ax5.axhline(y=expected_angle_between_runs, color='r', linestyle='dashed', label='expected')
    ax5.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # plot run durations
    ax6 = plt.subplot(rows, cols, 6)
    for agent_id, analysis in motility_analysis.items():
        run_duration = analysis['run_duration']
        run_time = analysis['run_time']
        mean_run_duration = np.mean(run_duration)
        ax6.scatter(run_time, run_duration, label=agent_id)
        ax6.axhline(y=mean_run_duration, linestyle='dashed')
    ax6.set_ylabel('run \n duration \n (s)')
    ax6.axhline(y=expected_run_duration, color='r', linestyle='dashed', label='expected')
    ax6.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # plot tumble durations
    ax7 = plt.subplot(rows, cols, 7)
    for agent_id, analysis in motility_analysis.items():
        tumble_duration = analysis['tumble_duration']
        tumble_time = analysis['tumble_time']
        mean_tumble_duration = np.mean(tumble_duration)
        ax7.scatter(tumble_time, tumble_duration, label=agent_id)
        ax7.axhline(y=mean_tumble_duration, linestyle='dashed')
    ax7.set_ylabel('tumble \n duration \n (s)')
    ax7.axhline(y=expected_tumble_duration, color='r', linestyle='dashed', label='expected')
    ax7.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig_path = os.path.join(out_dir, filename)
    plt.subplots_adjust(wspace=0.7, hspace=0.4)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)