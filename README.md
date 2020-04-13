## timed_feedforward_control

This repository contains code to run the simulations and create the figures for:

[Mohren, T. L., Daniel, T. L., & Brunton, S. L. (2020). Learning Precisely Timed Feedforward Control of the Sensor-Denied Inverted Pendulum. IEEE Control Systems Letters.](https://ieeexplore.ieee.org/abstract/document/9044302)


### Folder structure:
    simulation_scripts
        computeCostJ_filter_U3_variable_dt.ipynb    % runs code to find optimal control state space (fig1)
        pendulum_continuouslearning_dev.inpynb    % runs code for the online learning simulation (fig5)
        pendulum_find_feedforward_grouping_basinhopping.ipynb    % runs optimization for start-time and duration parameters
        phasediagram.py    % contains functions for plotting state space
        singlependulum    % contains functions for simulations
        u_(name) % contains data to be used by figure scripts
    figure_scripts
        figure(j)_(name).ipynb     % jupyter notebook to generate figure j
        latex_scientificPaperStyle.mplstyle     % general plotting parameters
        figure_functions.py    % contains functions for plotting
        figure_settings.py    % contains parameters for plotting
        phasediagram.py    % contains functions for plotting
        singlependulum.py    % contains functions for plotting
        u_(name)    % contains results from simulation scripts
        figs
            fig(j)_(name).(extension) % figure j in png, svg, pdf format
    README.md
